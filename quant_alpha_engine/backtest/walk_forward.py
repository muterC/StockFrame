"""
WalkForward — 滚动样本外验证（含 Purge / Embargo / Regime 分析）
================================================================

时序约定（无未来函数）
----------------------
       train (含 purge 末段)        embargo gap     test
    [t0 ........ train_end .....]  [...]  [test_start ........ test_end)

  - purge   : train 末尾扔掉 N 天（López de Prado purged CV 思想），
              主要给 ML 类 factor_fn 防 label-leakage 用
  - embargo : train 与 test 之间再隔 M 天 buffer，防序列自相关泄露
  - test_window 段是 OOS 评估区间，仅这段计入 fold 指标。

factor_fn 签名（自动识别两种约定）
----------------------------------
  ① 旧式 ：``factor_fn(close, **factor_kwargs) -> factor_df``
  ② 新式 ：``factor_fn(data,  **factor_kwargs) -> factor_df``
            其中 ``data`` 是 dict[str, DataFrame]，所有矩阵已按 ``[0, test_end]``
            一致切片，并附带 ``data["_train_end_idx"]`` 指示 ML 训练边界。

  框架通过 ``inspect.signature`` 检测第一个位置参数名：``data`` ⇒ 新式，
  其它 ⇒ 旧式（兼容老代码）。

典型用法
--------
>>> def my_factor(data, window=20):
...     close  = data["close"]
...     volume = data["volume"]
...     return op.Rank(op.Ts_Delta(close, window))
>>>
>>> wf = WalkForward(
...     factor_fn       = my_factor,
...     close           = close,
...     is_suspended    = is_susp,
...     is_limit        = is_limit,
...     data            = {"open": open_df, "volume": vol_df},
...     train_window    = 252,
...     test_window     = 63,
...     step            = 63,
...     purge           = 5,
...     embargo         = 2,
...     benchmark       = csi300_close,
...     factor_kwargs   = {"window": 20},
...     backtest_kwargs = dict(top_n=30, rebalance_freq=5, delay=1),
... )
>>> report = wf.run()
>>> report.print_summary()
>>> report.regime_metrics
>>> report.equity_curve.plot()
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant_alpha_engine.backtest.vector_engine import VectorEngine, BacktestResult
from quant_alpha_engine.backtest.performance import Performance


# ===========================================================================
# 结果容器
# ===========================================================================

@dataclass
class WalkForwardReport:
    """
    Walk-Forward 报告。

    Attributes
    ----------
    windows         : 每个窗口 (train_start, train_end, test_start, test_end) 的日期
    fold_metrics    : list[dict]，每个窗口的 OOS 指标
    summary         : 每窗指标拼成的 DataFrame
    equity_curve    : 全 OOS daily_returns 拼接得到的整体净值
    oos_sharpe_mean / oos_sharpe_std / oos_sharpe_winrate
    oos_ic_mean
    regime_metrics  : 若提供 benchmark，则按 Bull/Sideways/Bear 分组聚合
    skipped_folds   : 因 sanity check 跳过的 fold 列表
    fold_results    : 每窗的完整 BacktestResult
    """
    windows:             List[Tuple[Any, Any, Any, Any]]
    fold_metrics:        List[dict]
    summary:             pd.DataFrame
    equity_curve:        pd.Series
    oos_sharpe_mean:     float
    oos_sharpe_std:      float
    oos_sharpe_winrate:  float
    oos_ic_mean:         float
    regime_metrics:      Optional[Dict[str, dict]] = None
    skipped_folds:       List[int]                 = field(default_factory=list)
    fold_results:        List[BacktestResult]      = field(default_factory=list)

    def print_summary(self) -> None:
        print("=" * 64)
        print(f"Walk-Forward OOS 汇总  (folds={len(self.fold_metrics)}, "
              f"skipped={len(self.skipped_folds)})")
        print("=" * 64)
        print(f"  OOS Sharpe   均值 = {self.oos_sharpe_mean:+.3f}")
        print(f"  OOS Sharpe   标差 = {self.oos_sharpe_std:+.3f}")
        print(f"  OOS Sharpe   胜率 = {self.oos_sharpe_winrate * 100:.1f}%")
        print(f"  OOS IC       均值 = {self.oos_ic_mean:+.4f}")
        if self.regime_metrics:
            print("-" * 64)
            print("  Regime 分组（按基准同期收益 ±5%）")
            for regime, m in self.regime_metrics.items():
                print(f"    {regime:<10s}  folds={m['n_folds']:>2d}  "
                      f"Sharpe={m['Sharpe_mean']:+.2f}  "
                      f"OOS_累计={m['OOS_Cum_mean']*100:+.2f}%")
        print("=" * 64)
        with pd.option_context("display.max_rows", 50,
                               "display.width", 200,
                               "display.max_columns", 20):
            print(self.summary.to_string())


# ===========================================================================
# WalkForward 引擎
# ===========================================================================

class WalkForward:
    """
    滚动样本外验证（含 Purge / Embargo / Regime 分析）。

    参数说明
    --------
    factor_fn : Callable
        因子工厂函数，两种签名自动识别：
          - ``factor_fn(close, **kw)`` —— 旧式（兼容）
          - ``factor_fn(data,  **kw)`` —— 新式，data 为 dict
        新式 data 包含：
          - "close"、"is_suspended"、"is_limit"、用户 ``data`` 中的所有键
          - "_train_end_idx"  —— 本 fold 训练数据的有效末尾（已扣 purge）
          - "_train_end_date" —— 对应日期
        所有矩阵均按 [0, test_end] 一致切片。
    close, is_suspended, is_limit : T×N 全样本数据矩阵（必填）
    data                          : dict[name, DataFrame]，可选额外矩阵（open/volume/...）
    train_window  : 训练窗口长度（交易日）
    test_window   : 每个 OOS 测试窗口长度
    step          : 滑动步长，默认 = test_window（不重叠）
    purge         : train 末尾扔掉的天数，默认 0
    embargo       : train 与 test 之间的 buffer 天数，默认 0
    benchmark     : pd.Series，基准收盘价；用于 regime 分组（可选）
    regime_thresholds : (bear_th, bull_th)，默认 (-0.05, 0.05)
    factor_kwargs / backtest_kwargs : 透传给 factor_fn / VectorEngine
    sanity_check  : True 时若 OOS 区间因子全 NaN 则跳过该 fold（默认 True）
    """

    def __init__(
        self,
        factor_fn:       Callable[..., pd.DataFrame],
        close:           pd.DataFrame,
        is_suspended:    pd.DataFrame,
        is_limit:        pd.DataFrame,
        data:            Optional[Dict[str, pd.DataFrame]] = None,
        train_window:    int = 252,
        test_window:     int = 63,
        step:            Optional[int] = None,
        purge:           int = 0,
        embargo:         int = 0,
        benchmark:       Optional[pd.Series] = None,
        regime_thresholds: Tuple[float, float] = (-0.05, 0.05),
        factor_kwargs:   Optional[Dict] = None,
        backtest_kwargs: Optional[Dict] = None,
        sanity_check:    bool = True,
    ):
        if train_window < 20 or test_window < 5:
            raise ValueError("train_window/test_window 太小，建议 ≥ 20/5")
        if purge < 0 or embargo < 0:
            raise ValueError("purge / embargo 必须 ≥ 0")
        if regime_thresholds[0] >= regime_thresholds[1]:
            raise ValueError("regime_thresholds 必须满足 bear_th < bull_th")

        self.factor_fn         = factor_fn
        self.close             = close
        self.is_suspended      = is_suspended
        self.is_limit          = is_limit
        self.extra_data        = data or {}
        self.train_window      = int(train_window)
        self.test_window       = int(test_window)
        self.step              = int(step) if step else int(test_window)
        self.purge             = int(purge)
        self.embargo           = int(embargo)
        self.benchmark         = benchmark
        self.regime_thresholds = regime_thresholds
        self.factor_kwargs     = factor_kwargs or {}
        self.backtest_kwargs   = backtest_kwargs or {}
        self.sanity_check      = bool(sanity_check)

        # 自动识别 factor_fn 签名
        self._uses_dict_api = self._detect_dict_api(factor_fn)

    # ------------------------------------------------------------------

    @staticmethod
    def _detect_dict_api(factor_fn: Callable) -> bool:
        """根据 factor_fn 第一个位置参数名判断使用哪种契约。"""
        try:
            sig = inspect.signature(factor_fn)
            params = [p for p in sig.parameters.values()
                      if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                    inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            if not params:
                return False
            return params[0].name == "data"
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------

    def _make_windows(self, n_dates: int) -> List[Tuple[int, int, int, int]]:
        """
        生成 (train_start, train_end, test_start, test_end) 四元组（行索引）。

        train_end   = t - purge
        test_start  = t + embargo
        test_end    = test_start + test_window
        """
        windows: List[Tuple[int, int, int, int]] = []
        t = self.train_window
        while True:
            train_start = t - self.train_window
            train_end   = max(train_start + 1, t - self.purge)  # exclusive
            test_start  = t + self.embargo
            test_end    = test_start + self.test_window
            if test_end > n_dates:
                break
            windows.append((train_start, train_end, test_start, test_end))
            t += self.step
        return windows

    # ------------------------------------------------------------------

    def _build_data_slice(
        self,
        ve: int,
        train_end_idx: int,
        train_end_date,
    ) -> Dict[str, Any]:
        """构造统一切片的 data 字典（[0, ve] 切片）。"""
        out: Dict[str, Any] = {
            "close":           self.close.iloc[:ve],
            "is_suspended":    self.is_suspended.iloc[:ve],
            "is_limit":        self.is_limit.iloc[:ve],
            "_train_end_idx":  int(train_end_idx),
            "_train_end_date": train_end_date,
        }
        for name, df in self.extra_data.items():
            if df is None:
                out[name] = None
                continue
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"data['{name}'] 必须是 DataFrame，收到 {type(df)}")
            # 与 close 对齐索引后切片，避免长度不一致
            aligned = df.reindex(self.close.index[:ve])
            out[name] = aligned
        return out

    # ------------------------------------------------------------------

    def _classify_regime(
        self,
        test_start_d,
        test_end_d,
    ) -> Tuple[Optional[str], float]:
        """根据 benchmark 同期累计收益给 fold 打标签。"""
        if self.benchmark is None:
            return None, float("nan")
        bm = self.benchmark.loc[test_start_d:test_end_d].dropna()
        if len(bm) < 2:
            return "Unknown", float("nan")
        bm_ret = float(bm.iloc[-1] / bm.iloc[0] - 1)
        bear_th, bull_th = self.regime_thresholds
        if bm_ret <= bear_th:
            label = "Bear"
        elif bm_ret >= bull_th:
            label = "Bull"
        else:
            label = "Sideways"
        return label, bm_ret

    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> WalkForwardReport:
        dates = self.close.index
        n     = len(dates)
        windows = self._make_windows(n)
        if not windows:
            need = self.train_window + self.embargo + self.test_window
            raise ValueError(
                f"数据太短：n={n} < train_window+embargo+test_window={need}"
            )

        fold_metrics: List[dict] = []
        fold_results: List[BacktestResult] = []
        fold_returns: List[pd.Series]      = []
        window_dates: List[Tuple[Any, Any, Any, Any]] = []
        skipped: List[int] = []

        for k, (ts, te, vs, ve) in enumerate(windows, 1):
            test_start_d = dates[vs]
            test_end_d   = dates[ve - 1]
            train_end_d  = dates[te - 1]
            window_dates.append((dates[ts], train_end_d, test_start_d, test_end_d))

            # 1) 构造统一切片数据 → 调 factor_fn
            data_slice = self._build_data_slice(ve, te - 1, train_end_d)
            if self._uses_dict_api:
                factor = self.factor_fn(data_slice, **self.factor_kwargs)
            else:
                factor = self.factor_fn(data_slice["close"], **self.factor_kwargs)

            # 2) sanity check：OOS 段必须有有效因子值
            if self.sanity_check:
                try:
                    oos_factor = factor.loc[test_start_d:test_end_d]
                except (KeyError, TypeError):
                    oos_factor = (
                        factor.iloc[vs:ve] if len(factor) >= ve else factor.iloc[0:0]
                    )
                valid_cnt = int(oos_factor.notna().sum().sum()) if len(oos_factor) else 0
                if valid_cnt == 0:
                    warnings.warn(
                        f"[WalkForward] fold {k} OOS 区间 [{test_start_d.date()} ~ "
                        f"{test_end_d.date()}] 因子全为 NaN，跳过该 fold"
                    )
                    skipped.append(k)
                    continue

            # 3) OOS 段单独跑回测；其它矩阵也切片喂入，保持一致
            bt_kwargs = dict(self.backtest_kwargs)
            bt_kwargs["start_date"] = str(test_start_d.date())
            bt_kwargs["end_date"]   = str(test_end_d.date())

            engine = VectorEngine(
                factor       = factor,
                close        = self.close.iloc[:ve],
                is_suspended = self.is_suspended.iloc[:ve],
                is_limit     = self.is_limit.iloc[:ve],
                **bt_kwargs,
            )
            res = engine.run()

            # 4) OOS 累计 / 月化收益（替代误导性的"年化收益率"）
            if len(res.nav) >= 2 and res.nav.iloc[0] > 0:
                oos_cum = float(res.nav.iloc[-1] / res.nav.iloc[0] - 1)
                n_days  = len(res.nav)
                if n_days > 0 and (1 + oos_cum) > 0:
                    oos_monthly = (1 + oos_cum) ** (21.0 / n_days) - 1
                else:
                    oos_monthly = float("nan")
            else:
                oos_cum, oos_monthly = float("nan"), float("nan")

            # 5) MDD 详情
            mdd_info = Performance.calc_max_drawdown_detailed(res.nav)

            # 6) Regime 标签
            regime_label, bm_ret = self._classify_regime(test_start_d, test_end_d)

            row: Dict[str, Any] = {
                "fold":          k,
                "train_end":     train_end_d.date(),
                "test_start":    test_start_d.date(),
                "test_end":      test_end_d.date(),
                "Sharpe_Ratio":  res.metrics.get("Sharpe_Ratio", np.nan),
                "OOS_累计收益":   oos_cum,
                "OOS_月化收益":   oos_monthly,
                "最大回撤":       mdd_info["MaxDD"],
                "MDD_天数":      mdd_info["MaxDD_Duration"],
                "MDD_已修复":    mdd_info["MaxDD_Recovered"],
                "IC_Mean":       res.metrics.get("IC_Mean", np.nan),
                "ICIR":          res.metrics.get("ICIR", np.nan),
                "日均换手率":     res.metrics.get("日均换手率", np.nan),
            }
            if regime_label is not None:
                row["Regime"]    = regime_label
                row["Benchmark"] = bm_ret

            fold_metrics.append(row)
            fold_results.append(res)
            fold_returns.append(res.daily_returns)

            if verbose:
                tag = f" {regime_label}" if regime_label else ""
                print(f"[WalkForward] fold {k}/{len(windows)}  "
                      f"OOS [{test_start_d.date()} ~ {test_end_d.date()}]"
                      f"{tag}  Sharpe={row['Sharpe_Ratio']:+.2f}  "
                      f"Cum={oos_cum * 100:+.2f}%  IC={row['IC_Mean']:+.4f}")

        if not fold_metrics:
            raise RuntimeError(
                "所有 fold 都被 sanity check 跳过，请检查 factor_fn 输出。"
            )

        summary = pd.DataFrame(fold_metrics).set_index("fold")

        # 拼接全 OOS 净值（重叠日去重，后窗覆盖前窗）
        all_ret = pd.concat(fold_returns).sort_index()
        all_ret = all_ret[~all_ret.index.duplicated(keep="last")]
        equity  = (1 + all_ret.fillna(0.0)).cumprod()
        equity.name = "oos_equity"

        sharpe_series = summary["Sharpe_Ratio"].dropna()
        sharpe_mean   = float(sharpe_series.mean()) if len(sharpe_series) else float("nan")
        sharpe_std    = float(sharpe_series.std(ddof=1)) if len(sharpe_series) > 1 else 0.0
        winrate       = float((sharpe_series > 0).mean()) if len(sharpe_series) else float("nan")
        ic_mean       = (
            float(summary["IC_Mean"].dropna().mean())
            if "IC_Mean" in summary else float("nan")
        )

        # Regime 聚合
        regime_metrics: Optional[Dict[str, dict]] = None
        if "Regime" in summary.columns:
            regime_metrics = {}
            for regime, sub in summary.groupby("Regime"):
                regime_metrics[regime] = {
                    "n_folds":      int(len(sub)),
                    "Sharpe_mean":  float(sub["Sharpe_Ratio"].mean()),
                    "Sharpe_std":   float(sub["Sharpe_Ratio"].std(ddof=1)) if len(sub) > 1 else 0.0,
                    "OOS_Cum_mean": float(sub["OOS_累计收益"].mean()),
                    "IC_mean":      (
                        float(sub["IC_Mean"].mean()) if "IC_Mean" in sub else float("nan")
                    ),
                }

        return WalkForwardReport(
            windows            = window_dates,
            fold_metrics       = fold_metrics,
            summary            = summary,
            equity_curve       = equity,
            oos_sharpe_mean    = sharpe_mean,
            oos_sharpe_std     = sharpe_std,
            oos_sharpe_winrate = winrate,
            oos_ic_mean        = ic_mean,
            regime_metrics     = regime_metrics,
            skipped_folds      = skipped,
            fold_results       = fold_results,
        )
