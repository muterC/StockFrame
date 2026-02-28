"""
tuning/tuner.py
===============
ParameterTuner — 参数调优引擎

设计概述
--------
对「因子工厂函数参数」×「VectorEngine 回测参数」做笛卡尔积搜索：

1. 根据 ``factor_params`` 和 ``backtest_params`` 生成所有参数组合
2. 对每个组合调用用户提供的 ``factor_fn`` 构造因子，再通过 ``VectorEngine`` 执行回测
3. 按 ``opt_target`` 定义的多指标优先级排序结果
4. 提供 ``top(n)``、``get_result(idx)``、``summary()`` 等接口方便分析

防未来函数说明
--------------
- 因子参数（window 等）影响因子计算，不产生未来函数
- 回测参数均为 VectorEngine 的合法 kwargs，由引擎内部保证时序安全
- 建议始终保持 ``delay >= 1``（VectorEngine 默认值）

Quick Start::

    from quant_alpha_engine.tuning import ParameterTuner
    from quant_alpha_engine.ops import AlphaOps as op

    def my_factor(close, volume, high, low, window=20, vol_window=10):
        return op.Rank(op.RiskAdjMomentum(close, window=window, vol_window=vol_window))

    tuner = ParameterTuner(
        factor_fn       = my_factor,
        close           = close,
        is_suspended    = is_susp,
        is_limit        = is_limit,
        factor_params   = {'window': [10, 20, 30], 'vol_window': [5, 10]},
        backtest_params = {'rebalance_freq': [1, 5], 'top_n': [20, 30]},
        opt_target      = {'Sharpe_Ratio': 'desc', '最大回撤': 'asc'},
        n_jobs          = 4,
    )
    tuner.run()
    tuner.print_top(5)
    tuner.get_result(0).plot()
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from quant_alpha_engine.backtest.vector_engine import BacktestResult, VectorEngine

# ── tqdm (可选) ─────────────────────────────────────────────────────────────
try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


# ── 常量 ────────────────────────────────────────────────────────────────────

# VectorEngine 接受的合法回测参数键
_VALID_BT_PARAMS = frozenset([
    "rebalance_freq", "top_n", "weight_method",
    "cost_rate", "delay", "decay",
    "start_date", "end_date",
])

# 结果 DataFrame 中展示的指标列（按顺序）
_METRIC_COLS = [
    "年化收益率", "Sharpe_Ratio", "最大回撤",
    "IC_Mean", "ICIR", "IC_胜率",
    "年化波动率", "Calmar_Ratio", "IC_Std", "Fitness",
]

# Styler 颜色配置（与 vector_engine._print_metrics_table 保持一致）
_METRIC_COLORS: Dict[str, dict] = {
    "年化收益率":  {"lower_is_better": False,
                   "bands": [(0.15, "#c0392b"), (0.25, "#2980b9"), (None, "#27ae60")]},
    "Sharpe_Ratio": {"lower_is_better": False,
                     "bands": [(1.0, "#c0392b"), (1.5, "#2980b9"), (None, "#27ae60")]},
    "最大回撤":    {"lower_is_better": True,
                   "bands": [(0.05, "#27ae60"), (0.10, "#2980b9"), (None, "#c0392b")]},
    "IC_Mean":     {"lower_is_better": False,
                   "bands": [(0.03, "#c0392b"), (0.05, "#2980b9"), (None, "#27ae60")]},
    "ICIR":        {"lower_is_better": False,
                   "bands": [(1.5, "#c0392b"), (2.5, "#2980b9"), (None, "#27ae60")]},
    "IC_胜率":     {"lower_is_better": False,
                   "bands": [(0.55, "#c0392b"), (0.60, "#2980b9"), (None, "#27ae60")]},
    "年化波动率":  {"lower_is_better": True,
                   "bands": [(0.10, "#27ae60"), (0.20, "#2980b9"), (0.30, "#c0392b"), (None, "#8e44ad")]},
    "Calmar_Ratio": {"lower_is_better": False,
                     "bands": [(0.5, "#c0392b"), (1.0, "#d4ac0d"), (2.0, "#2980b9"), (None, "#27ae60")]},
    "IC_Std":      {"lower_is_better": True,
                   "bands": [(0.10, "#27ae60"), (0.15, "#2980b9"), (0.20, "#d4ac0d"), (None, "#8e44ad")]},
    "Fitness":     {"lower_is_better": False,
                   "bands": [(0.3, "#c0392b"), (0.6, "#2980b9"), (None, "#27ae60")]},
}


# ============================================================================
# 内部辅助函数
# ============================================================================

def _make_combo_id(fp: dict, bp: dict) -> str:
    """生成简洁的参数组合标识符字符串。"""
    parts = []
    for k, v in sorted(fp.items()):
        parts.append(f"{k}={v}")
    for k, v in sorted(bp.items()):
        parts.append(f"{k}={v}")
    return "|".join(parts)


def _build_combos(
    factor_params: Dict[str, list],
    backtest_params: Dict[str, list],
) -> List[dict]:
    """
    生成因子参数 × 回测参数的笛卡尔积组合列表。

    Returns
    -------
    list of dict, each with keys:
        'id'              : str   — 唯一标识符
        'factor_params'   : dict  — 传给 factor_fn 的参数
        'backtest_params' : dict  — 传给 VectorEngine 的参数
    """
    # 合并两个参数空间，用前缀区分
    all_keys: List[str] = []
    all_values: List[list] = []

    fp_keys = sorted(factor_params.keys())
    bp_keys = sorted(backtest_params.keys())

    for k in fp_keys:
        all_keys.append(f"__fp__{k}")
        all_values.append(factor_params[k])

    for k in bp_keys:
        all_keys.append(f"__bp__{k}")
        all_values.append(backtest_params[k])

    # 空参数字典处理：笛卡尔积仍正确生成单一空组合
    if not all_values:
        return [{"id": "default", "factor_params": {}, "backtest_params": {}}]

    combos = []
    for values in product(*all_values):
        mapping = dict(zip(all_keys, values))
        fp = {k[6:]: v for k, v in mapping.items() if k.startswith("__fp__")}
        bp = {k[6:]: v for k, v in mapping.items() if k.startswith("__bp__")}
        combos.append({
            "id": _make_combo_id(fp, bp),
            "factor_params": fp,
            "backtest_params": bp,
        })
    return combos


def _get_color(metric: str, val: float) -> str:
    """根据指标值返回对应的背景颜色十六进制字符串。"""
    if metric not in _METRIC_COLORS or (val != val):  # NaN check
        return ""
    cfg = _METRIC_COLORS[metric]
    bands = cfg["bands"]
    lower_is_better = cfg["lower_is_better"]
    abs_val = abs(val)  # 最大回撤等为负值，取绝对值比较

    if not lower_is_better:
        # 从最低档向上找
        color = bands[0][1]  # 默认最差
        for upper, clr in bands:
            if upper is None:
                color = clr
                break
            if abs_val < upper:
                break
            color = clr
        return color
    else:
        # lower_is_better: 值越小越好，从小往大找
        for upper, clr in bands:
            if upper is None:
                return clr
            if abs_val <= upper:
                return clr
        return bands[-1][1]


def _fmt_val(metric: str, val: float) -> str:
    """格式化单个指标值为显示字符串。"""
    if val != val:  # NaN
        return "N/A"
    pct_metrics = {"年化收益率", "最大回撤", "IC_胜率", "年化波动率"}
    if metric in pct_metrics:
        return f"{val * 100:+.2f}%"
    return f"{val:+.4f}"


def _build_styled_df(df: pd.DataFrame, metric_cols: List[str]) -> "pd.io.formats.style.Styler":
    """
    对 DataFrame 中的指标列按颜色分级着色，返回 pandas Styler 对象。
    非指标列（参数列）保持默认样式。
    """
    # 格式化显示值
    fmt_dict = {}
    for col in metric_cols:
        if col in df.columns:
            fmt_dict[col] = lambda v, m=col: _fmt_val(m, v)

    def _style_cell(val, metric):
        color = _get_color(metric, val)
        if color:
            return f"background-color: {color}; color: white; font-weight: bold;"
        return ""

    styler = df.style

    # 应用颜色
    for col in metric_cols:
        if col in df.columns:
            styler = styler.applymap(
                lambda v, m=col: _style_cell(v, m),
                subset=[col],
            )

    # 格式化数字显示
    for col in metric_cols:
        if col in df.columns:
            styler = styler.format(
                {col: lambda v, m=col: _fmt_val(m, v)}
            )

    # 整体表格样式
    styler = styler.set_table_styles([
        {"selector": "th",
         "props": [("background-color", "#1a1a2e"),
                   ("color", "#ecf0f1"),
                   ("font-size", "12px"),
                   ("padding", "6px 12px"),
                   ("text-align", "center")]},
        {"selector": "td",
         "props": [("font-size", "11px"),
                   ("padding", "5px 10px"),
                   ("text-align", "center")]},
        {"selector": "tr:nth-child(even) td",
         "props": [("background-color", "#f8f9fa")]},
        {"selector": "caption",
         "props": [("font-size", "13px"),
                   ("font-weight", "bold"),
                   ("padding-bottom", "8px")]},
    ])

    return styler


# ============================================================================
# ParameterTuner 主类
# ============================================================================

class ParameterTuner:
    """
    参数调优引擎。

    对「因子工厂参数」和「VectorEngine 回测参数」进行笛卡尔积搜索，
    自动执行所有组合的回测，并按指定的多目标指标排序结果。

    Parameters
    ----------
    factor_fn : Callable
        因子工厂函数，签名为::

            factor_fn(close, volume, high, low, **factor_params) -> pd.DataFrame

        函数接收价格/量数据 + 因子参数，返回 (T×N) 因子矩阵。

    close : pd.DataFrame
        收盘价矩阵 (T×N)。

    is_suspended : pd.DataFrame
        停牌标记矩阵 (T×N)，True = 当日停牌。

    is_limit : pd.DataFrame
        涨跌停标记矩阵 (T×N)，True = 当日涨/跌停。

    factor_params : dict of {str: list}
        因子参数搜索空间，例如 ``{'window': [10, 20, 30]}``.

    backtest_params : dict of {str: list}
        回测参数搜索空间，合法键为 VectorEngine 接受的参数：
        ``rebalance_freq``, ``top_n``, ``weight_method``,
        ``cost_rate``, ``delay``, ``decay``, ``start_date``, ``end_date``.

    opt_target : dict of {str: str}
        多目标优化方向，键为指标名（同 BacktestResult.metrics 中的键），
        值为 ``'desc'``（越大越好）或 ``'asc'``（越小越好）。
        多个键时，按字典顺序决定排序优先级（第一键优先）。
        例如 ``{'Sharpe_Ratio': 'desc', '最大回撤': 'asc'}``.

    volume : pd.DataFrame, optional
        成交量矩阵 (T×N)，若因子工厂函数需要则传入。

    high : pd.DataFrame, optional
        最高价矩阵 (T×N)。

    low : pd.DataFrame, optional
        最低价矩阵 (T×N)。

    industry : pd.Series or pd.DataFrame, optional
        行业分类，传入后 VectorEngine 自动做 OLS 行业中性化。

    n_jobs : int, default 1
        并行线程数。``1`` = 顺序执行；``>1`` = 使用 ThreadPoolExecutor 并行。
        注意：并行时进度条顺序可能与提交顺序不同。

    verbose : bool, default True
        是否打印进度信息。

    Attributes
    ----------
    combos : list of dict
        所有参数组合列表（调用 run() 前也可访问）。

    results_df : pd.DataFrame
        回测完成后的完整结果表（按 opt_target 排序）。
        调用 run() 前为 None。

    Examples
    --------
    >>> from quant_alpha_engine.tuning import ParameterTuner
    >>> from quant_alpha_engine.ops import AlphaOps as op
    >>>
    >>> def my_factor(close, volume, high, low, window=20):
    ...     return op.Rank(op.Ts_Delta(close, window))
    >>>
    >>> tuner = ParameterTuner(
    ...     factor_fn=my_factor,
    ...     close=close, is_suspended=is_susp, is_limit=is_limit,
    ...     factor_params={'window': [10, 20, 30]},
    ...     backtest_params={'rebalance_freq': [1, 5], 'top_n': [30]},
    ...     opt_target={'Sharpe_Ratio': 'desc'},
    ...     n_jobs=4,
    ... )
    >>> tuner.run()
    >>> tuner.print_top(3)
    >>> tuner.get_result(0).plot()
    """

    def __init__(
        self,
        factor_fn: Callable,
        close: pd.DataFrame,
        is_suspended: pd.DataFrame,
        is_limit: pd.DataFrame,
        factor_params: Dict[str, list],
        backtest_params: Dict[str, list],
        opt_target: Dict[str, str],
        volume: Optional[pd.DataFrame] = None,
        high: Optional[pd.DataFrame] = None,
        low: Optional[pd.DataFrame] = None,
        industry: Optional[Union[pd.Series, pd.DataFrame]] = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> None:
        # ── 参数校验 ──────────────────────────────────────────────────
        if not callable(factor_fn):
            raise TypeError(f"factor_fn 必须是可调用对象，当前类型: {type(factor_fn).__name__}")
        if not isinstance(close, pd.DataFrame) or close.empty:
            raise ValueError("close 必须是非空 pd.DataFrame")
        if not isinstance(is_suspended, pd.DataFrame) or is_suspended.empty:
            raise ValueError("is_suspended 必须是非空 pd.DataFrame")
        if not isinstance(is_limit, pd.DataFrame) or is_limit.empty:
            raise ValueError("is_limit 必须是非空 pd.DataFrame")
        if not isinstance(factor_params, dict):
            raise TypeError(f"factor_params 必须是 dict，当前类型: {type(factor_params).__name__}")
        if not isinstance(backtest_params, dict):
            raise TypeError(f"backtest_params 必须是 dict，当前类型: {type(backtest_params).__name__}")
        if not isinstance(opt_target, dict) or len(opt_target) == 0:
            raise ValueError("opt_target 必须是非空 dict，如 {'Sharpe_Ratio': 'desc'}")

        # 校验 backtest_params 键合法性
        invalid_bp = set(backtest_params.keys()) - _VALID_BT_PARAMS
        if invalid_bp:
            raise ValueError(
                f"backtest_params 包含非法键: {sorted(invalid_bp)}。"
                f"合法键为: {sorted(_VALID_BT_PARAMS)}"
            )

        # 校验 opt_target 方向值
        for k, v in opt_target.items():
            if v not in ("asc", "desc"):
                raise ValueError(
                    f"opt_target['{k}'] = '{v}' 不合法，必须是 'asc' 或 'desc'"
                )

        if not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError(f"n_jobs 必须是正整数，当前值: {n_jobs!r}")

        # ── 存储属性 ───────────────────────────────────────────────────
        self._factor_fn = factor_fn
        self._close = close
        self._is_suspended = is_suspended
        self._is_limit = is_limit
        self._factor_params = factor_params
        self._backtest_params = backtest_params
        self._opt_target = opt_target
        self._volume = volume
        self._high = high
        self._low = low
        self._industry = industry
        self._n_jobs = n_jobs
        self._verbose = verbose

        # ── 生成笛卡尔积参数组合 ───────────────────────────────────────
        self._combos: List[dict] = _build_combos(factor_params, backtest_params)

        # ── 结果存储 ───────────────────────────────────────────────────
        self._results_df: Optional[pd.DataFrame] = None
        self._bt_results: Dict[str, BacktestResult] = {}  # combo_id -> BacktestResult
        self._ranked_ids: List[str] = []  # 排序后的 combo_id 列表

    # ──────────────────────────────────────────────────────────────────────
    # 属性
    # ──────────────────────────────────────────────────────────────────────

    @property
    def combos(self) -> List[dict]:
        """所有参数组合列表（调用 run() 前也可访问）。"""
        return self._combos

    @property
    def results_df(self) -> Optional[pd.DataFrame]:
        """
        回测完成后的完整结果表（按 opt_target 排序，含 Rank 列）。
        调用 run() 前为 None。
        """
        return self._results_df

    # ──────────────────────────────────────────────────────────────────────
    # 核心方法
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> "ParameterTuner":
        """
        执行所有参数组合的回测。

        遍历笛卡尔积组合，依次（或并行）调用因子工厂 + VectorEngine，
        收集结果并按 opt_target 排序。

        Returns
        -------
        self
            方便链式调用：``tuner.run().print_top(5)``

        Notes
        -----
        - 若某个组合回测失败，将记录警告并将该组合的指标置为 NaN，继续后续组合
        - 所有组合均失败时抛出 RuntimeError
        """
        n = len(self._combos)
        if self._verbose:
            print(f"[ParameterTuner] 开始调优：共 {n} 个参数组合"
                  f"，n_jobs={self._n_jobs}")

        self._bt_results = {}
        rows: List[dict] = []

        # ── 执行回测 ───────────────────────────────────────────────────
        if self._n_jobs == 1:
            # 顺序执行
            iterator = self._combos
            if self._verbose and _TQDM_AVAILABLE:
                iterator = _tqdm(iterator, desc="调优进度", unit="combo")
            for combo in iterator:
                row = self._run_single(combo)
                rows.append(row)
        else:
            # 多线程并行执行
            rows_map: Dict[str, dict] = {}
            with ThreadPoolExecutor(max_workers=self._n_jobs) as executor:
                future_to_combo = {
                    executor.submit(self._run_single, combo): combo
                    for combo in self._combos
                }
                completed_iter = as_completed(future_to_combo)
                if self._verbose and _TQDM_AVAILABLE:
                    completed_iter = _tqdm(
                        completed_iter, total=n, desc="调优进度", unit="combo"
                    )
                elif self._verbose:
                    completed_count = 0

                for future in completed_iter:
                    row = future.result()
                    rows_map[row["combo_id"]] = row
                    if self._verbose and not _TQDM_AVAILABLE:
                        completed_count += 1
                        print(f"  [{completed_count}/{n}] {row['combo_id']}")

            # 保持原始顺序（与 combos 一致）
            for combo in self._combos:
                rows.append(rows_map[combo["id"]])

        # ── 检查是否全部失败 ───────────────────────────────────────────
        n_success = sum(1 for r in rows if not r.get("_failed", False))
        if n_success == 0:
            raise RuntimeError(
                "所有参数组合均回测失败，请检查因子工厂函数和参数设置。"
            )
        if n_success < n and self._verbose:
            print(f"[ParameterTuner] 警告：{n - n_success} 个组合失败，"
                  f"已用 NaN 填充对应指标。")

        # ── 构建结果 DataFrame ─────────────────────────────────────────
        df = pd.DataFrame(rows)
        # 移除内部标记列
        if "_failed" in df.columns:
            df = df.drop(columns=["_failed"])

        # ── 多目标排序 ─────────────────────────────────────────────────
        sort_cols = [c for c in self._opt_target.keys() if c in df.columns]
        ascending = [self._opt_target[c] == "asc" for c in sort_cols]

        if sort_cols:
            df = df.sort_values(
                by=sort_cols,
                ascending=ascending,
                na_position="last",
            ).reset_index(drop=True)

        # 插入 Rank 列
        df.insert(0, "Rank", range(1, len(df) + 1))

        self._results_df = df
        self._ranked_ids = df["combo_id"].tolist()

        if self._verbose:
            print(f"[ParameterTuner] 调优完成！成功: {n_success}/{n} 个组合")
            if sort_cols:
                best_row = df.iloc[0]
                top_metrics = {c: best_row.get(c, float("nan")) for c in sort_cols}
                metrics_str = "  ".join(
                    f"{k}={_fmt_val(k, v)}" for k, v in top_metrics.items()
                )
                print(f"  Best Combo: {best_row['combo_id']}")
                print(f"  Best Metrics: {metrics_str}")

        return self

    def _run_single(self, combo: dict) -> dict:
        """
        执行单个参数组合的回测，返回结果行字典。

        Parameters
        ----------
        combo : dict
            包含 'id'、'factor_params'、'backtest_params' 的字典。

        Returns
        -------
        dict
            包含 combo_id、参数值、指标值的行字典。
            若回测失败，指标列为 NaN，并标记 '_failed=True'。
        """
        combo_id = combo["id"]
        fp = combo["factor_params"]
        bp = combo["backtest_params"]

        # 初始化行，包含所有参数列
        row: dict = {"combo_id": combo_id}

        # 添加因子参数列（前缀 f_）
        for k, v in fp.items():
            row[f"f_{k}"] = v

        # 添加回测参数列（前缀 b_）
        for k, v in bp.items():
            row[f"b_{k}"] = v

        try:
            # Step 1: 构造因子
            factor = self._factor_fn(
                self._close,
                self._volume,
                self._high,
                self._low,
                **fp,
            )

            # Step 2: 执行回测
            result = VectorEngine(
                factor=factor,
                close=self._close,
                is_suspended=self._is_suspended,
                is_limit=self._is_limit,
                industry=self._industry,
                **bp,
            ).run()

            # Step 3: 存储 BacktestResult
            self._bt_results[combo_id] = result

            # Step 4: 提取指标
            for metric in _METRIC_COLS:
                row[metric] = result.metrics.get(metric, float("nan"))

            row["_failed"] = False

        except Exception as exc:
            warnings.warn(
                f"[ParameterTuner] 组合 '{combo_id}' 回测失败: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
            for metric in _METRIC_COLS:
                row[metric] = float("nan")
            row["_failed"] = True

        return row

    # ──────────────────────────────────────────────────────────────────────
    # 结果查询接口
    # ──────────────────────────────────────────────────────────────────────

    def _check_run(self) -> None:
        """内部：检查 run() 是否已调用。"""
        if self._results_df is None:
            raise RuntimeError(
                "请先调用 run() 执行回测，再查询结果。"
            )

    def top(self, n: int = 5) -> pd.DataFrame:
        """
        返回 Top-N 最优组合的结果表。

        Parameters
        ----------
        n : int, default 5
            返回前 n 个排名最高的组合。

        Returns
        -------
        pd.DataFrame
            Top-N 结果表，仅含关键指标列。
        """
        self._check_run()
        n = min(n, len(self._results_df))
        return self._results_df.head(n).copy()

    def get_result(self, idx: int = 0) -> BacktestResult:
        """
        按排名索引（0-based）获取对应组合的完整 BacktestResult 对象。

        Parameters
        ----------
        idx : int, default 0
            排名索引，0 = 最优组合，1 = 第二优，以此类推。

        Returns
        -------
        BacktestResult
            对应组合的完整回测结果，支持 ``.plot()`` / ``.print_summary()``。

        Raises
        ------
        IndexError
            若 idx 超出范围。
        KeyError
            若对应组合回测失败（指标为 NaN），无法获取 BacktestResult。

        Examples
        --------
        >>> best = tuner.get_result(0)
        >>> best.print_summary()
        >>> best.plot()
        """
        self._check_run()
        if idx < 0 or idx >= len(self._ranked_ids):
            raise IndexError(
                f"idx={idx} 超出范围 [0, {len(self._ranked_ids) - 1}]"
            )
        combo_id = self._ranked_ids[idx]
        if combo_id not in self._bt_results:
            raise KeyError(
                f"Rank {idx + 1} 的组合 '{combo_id}' 回测失败，"
                "无法获取 BacktestResult。"
            )
        return self._bt_results[combo_id]

    def get_result_by_id(self, combo_id: str) -> BacktestResult:
        """
        通过 combo_id 字符串精确获取对应的 BacktestResult。

        Parameters
        ----------
        combo_id : str
            组合标识符（可从 results_df['combo_id'] 列查看）。

        Returns
        -------
        BacktestResult

        Examples
        --------
        >>> cid = tuner.results_df.iloc[0]['combo_id']
        >>> result = tuner.get_result_by_id(cid)
        """
        self._check_run()
        if combo_id not in self._bt_results:
            available = list(self._bt_results.keys())[:5]
            raise KeyError(
                f"未找到 combo_id='{combo_id}'。"
                f"可用的部分 ID: {available}"
            )
        return self._bt_results[combo_id]

    def summary(self) -> pd.DataFrame:
        """
        返回所有组合的完整排序结果表。

        Returns
        -------
        pd.DataFrame
            含 Rank、combo_id、参数列、指标列的完整 DataFrame。
        """
        self._check_run()
        return self._results_df.copy()

    # ──────────────────────────────────────────────────────────────────────
    # 显示接口
    # ──────────────────────────────────────────────────────────────────────

    def print_top(self, n: int = 5) -> None:
        """
        展示 Top-N 最优组合（Jupyter 中渲染为彩色 HTML 表格，终端回退为纯文本）。

        Parameters
        ----------
        n : int, default 5
            展示前 n 个组合。
        """
        self._check_run()
        n = min(n, len(self._results_df))
        df_top = self._results_df.head(n).copy()

        # 找出参数列和指标列
        all_cols = list(df_top.columns)
        metric_cols = [c for c in _METRIC_COLS if c in all_cols]

        # ── Jupyter 渲染 ───────────────────────────────────────────────
        try:
            from IPython.display import display, HTML

            # 构造 Styler
            styler = _build_styled_df(df_top, metric_cols)
            styler = styler.set_caption(
                f"Parameter Tuning Results — Top {n} "
                f"（排序依据: {', '.join(self._opt_target.keys())}）"
            )
            display(styler)
            return

        except Exception:
            pass

        # ── 终端纯文本回退 ─────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"  Parameter Tuning — Top {n} Results")
        print(f"  排序依据: {self._opt_target}")
        print(f"{'='*70}")

        # 格式化显示列
        display_cols = ["Rank", "combo_id"] + metric_cols
        display_cols = [c for c in display_cols if c in df_top.columns]
        df_show = df_top[display_cols].copy()

        for col in metric_cols:
            if col in df_show.columns:
                df_show[col] = df_show[col].apply(lambda v: _fmt_val(col, v))

        print(df_show.to_string(index=False))
        print(f"{'='*70}\n")

    def __repr__(self) -> str:
        n_combos = len(self._combos)
        n_done = len(self._bt_results)
        status = f"{n_done}/{n_combos} 完成" if n_done > 0 else "未运行"
        return (
            f"ParameterTuner("
            f"combos={n_combos}, "
            f"status={status}, "
            f"opt={list(self._opt_target.keys())})"
        )
