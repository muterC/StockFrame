"""
VectorEngine — 矩阵式净值回测引擎
====================================
基于因子权重矩阵的向量化回测，严格防止未来函数（Look-ahead bias）。

时序语义（delay 参数的精确定义）
--------------------------------
  delay=d 表示：用 T-d 日的因子值预测 T 日的收益。

  以 delay=1 为例（标准配置）：
    - factor[T-1]  →  构建 T 日持仓权重
    - ret[T]       =  close[T] / close[T-1] - 1（T 日当日收益）
    - 含义：T-1 日收盘计算因子 → T 日开盘执行交易 → 赚 T 日收益

  以 delay=0 为例（仅回测研究用，实盘不可用）：
    - factor[T]    →  构建 T 日持仓权重
    - ret[T]       =  close[T] / close[T-1] - 1（T 日当日收益）
    - 含义：因子与收益处于同一天，存在 look-ahead bias

回测逻辑
--------
1. 数据对齐（factor / close / is_suspended / is_limit 取交集，可选 start_date/end_date 裁剪）
2. 因子预处理（delay → decay → 行业中性化）
3. 计算当日收益率 ret[T] = close[T] / close[T-1] - 1
4. 按调仓频率生成持仓权重矩阵（Top-N 等权 or 因子加权）
5. 过滤停牌/涨跌停股票（权重置零并重新归一化）
6. 扣除单边交易成本（手续费 + 滑点）
7. 输出净值序列及全量绩效指标

防止未来函数原则
--------------
- delay=1（默认）：factor 经 Ts_Delay(1) 后，T 行因子值来自 T-1 日
- 权重 weights[T] 由 factor[T-1] 构建（delay=1 保证）
- 收益 ret[T] = close[T]/close[T-1] - 1（当日收益，与权重同行对齐）
- 组合收益：port_ret[T] = sum_i(weights[T,i] * ret[T,i])

回测时间范围控制
--------------
- start_date / end_date 参数均为可选，格式为 'YYYY-MM-DD' 字符串或 None
- 裁剪发生在数据对齐之后，不影响因子计算的历史窗口（因子已提前计算好）
- 仅控制回测评估区间，适合做不同时间段的分段测试
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal, Union

import numpy as np
import pandas as pd

from quant_alpha_engine.backtest.performance import Performance
from quant_alpha_engine.visualization.report import Report


@dataclass
class BacktestResult:
    """
    回测结果容器。

    Attributes
    ----------
    nav           : 策略净值序列（从 1.0 开始）
    daily_returns : 扣除成本后的日收益率
    gross_returns : 扣除成本前的日收益率
    cost_series   : 每日交易成本
    weights       : 持仓权重矩阵 (T × N)
    turnover      : 每日单边换手率
    ic_series     : 每日截面 Rank IC
    forward_returns: 前向 1 日收益率矩阵
    factor        : 输入因子矩阵（对齐后）
    metrics       : 所有绩效指标字典
    rebalance_dates: 实际调仓日期列表
    """
    nav:             pd.Series
    daily_returns:   pd.Series
    gross_returns:   pd.Series
    cost_series:     pd.Series
    weights:         pd.DataFrame
    turnover:        pd.Series
    ic_series:       pd.Series
    forward_returns: pd.DataFrame
    factor:          pd.DataFrame
    metrics:         dict
    rebalance_dates: list = field(default_factory=list)

    def print_summary(self) -> None:
        """在控制台打印 Unicode 格式绩效报告。"""
        _print_metrics_table(self.metrics)

    def plot(self, save_path: Optional[str] = None) -> None:
        """
        绘制 6 子图回测分析报告。

        Parameters
        ----------
        save_path : 若指定路径，则保存为 PNG 文件；否则弹窗显示
        """
        Report.plot(self, save_path=save_path)


# ===========================================================================
# 核心回测引擎
# ===========================================================================

class VectorEngine:
    """
    矩阵式净值回测引擎。

    Parameters
    ----------
    factor          : 因子矩阵 (T × N)，index=日期，columns=股票代码
    close           : 收盘价矩阵 (T × N)
    is_suspended    : 停牌状态矩阵 (T × N, bool)，True 表示停牌
    is_limit        : 涨跌停状态矩阵 (T × N, bool)，True 表示触发涨跌停
    rebalance_freq  : 调仓频率（天数），默认 1（每日调仓）
    top_n           : 持仓股票数量，默认 50
    weight_method   : 权重计算方式，'equal'（等权）或 'factor_weighted'（因子值加权）
    cost_rate       : 单边交易成本（手续费+滑点），默认 0.0015（0.15%）
    initial_capital : 初始资金（仅用于展示，不影响收益率计算）
    delay           : 因子延迟天数，默认 1。
                      delay=d 表示用 T-d 日因子预测 T 日收益。
                      delay=1（推荐）：T-1 日因子 → T 日建仓 → 赚 T 日收益，严格无未来函数。
                      delay=0：T 日因子预测 T 日收益，存在 look-ahead bias，仅供研究。
    decay           : 线性衰减窗口大小，默认 0（不衰减）；>0 时对因子做 Decay_Linear
    industry        : 行业映射（pd.Series 或 pd.DataFrame），用于因子行业中性化；
                      None（默认）表示跳过中性化
    start_date      : 回测开始日期（含），格式 'YYYY-MM-DD' 或 None（不裁剪）。
                      裁剪发生在数据对齐之后，因子已提前计算好，不影响历史窗口。
                      例如：'2023-01-01' 表示仅对 2023年1月1日之后的数据进行回测评估。
    end_date        : 回测结束日期（含），格式 'YYYY-MM-DD' 或 None（不裁剪）。
                      例如：'2024-12-31' 表示仅对 2024年12月31日之前的数据进行回测评估。

    预处理执行顺序（均在数据对齐之后、权重构建之前）
    --------------------------------------------------
    1. delay  ：Ts_Delay(factor, delay)   — delay=1 使 factor[T] 变为原 factor[T-1]
    2. decay  ：Decay_Linear(factor, decay)
    3. 行业中性化：Neutralize(factor, industry)

    Examples
    --------
    >>> engine = VectorEngine(
    ...     factor=factor_df,
    ...     close=close_df,
    ...     is_suspended=suspended_df,
    ...     is_limit=limit_df,
    ...     rebalance_freq=5,
    ...     top_n=30,
    ...     delay=1,
    ...     decay=5,
    ...     industry=industry_series,
    ...     start_date='2023-01-01',
    ...     end_date='2024-12-31',
    ... )
    >>> result = engine.run()
    >>> result.print_summary()
    >>> result.plot()
    """

    def __init__(
        self,
        factor:          pd.DataFrame,
        close:           pd.DataFrame,
        is_suspended:    pd.DataFrame,
        is_limit:        pd.DataFrame,
        rebalance_freq:  int = 1,
        top_n:           int = 50,
        weight_method:   Literal["equal", "factor_weighted"] = "equal",
        cost_rate:       float = 0.0015,
        initial_capital: float = 1_000_000.0,
        # ── 因子预处理参数 ─────────────────────────────────────
        delay:    int                                        = 1,
        decay:    int                                        = 0,
        industry: Optional[Union[pd.Series, pd.DataFrame]]  = None,
        # ── 回测时间范围控制 ───────────────────────────────────
        start_date: Optional[str]                           = None,
        end_date:   Optional[str]                           = None,
    ):
        self.factor         = factor
        self.close          = close
        self.is_suspended   = is_suspended
        self.is_limit       = is_limit
        self.rebalance_freq = rebalance_freq
        self.top_n          = top_n
        self.weight_method  = weight_method
        self.cost_rate      = cost_rate
        self.initial_capital = initial_capital
        self.delay          = delay
        self.decay          = decay
        self.industry       = industry
        self.start_date     = start_date
        self.end_date       = end_date

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """
        执行完整回测流程。

        Returns
        -------
        BacktestResult : 包含净值、收益率、权重、IC、指标等全部结果
        """
        print("[VectorEngine] 开始对齐数据...")
        factor, close, is_suspended, is_limit = self._align_data()

        # 打印时间范围信息
        date_range_info = f"{factor.index[0].strftime('%Y-%m-%d')} ~ {factor.index[-1].strftime('%Y-%m-%d')}"
        print(f"[VectorEngine] 数据规模：{len(factor)} 个交易日 × {factor.shape[1]} 只股票  [{date_range_info}]")
        print(f"[VectorEngine] 预处理参数：delay={self.delay}, decay={self.decay}, "
              f"neutralize={'是' if self.industry is not None else '否'}")

        # Step 0: 因子预处理（delay → decay → 行业中性化）
        # delay=1 后：factor 矩阵第 T 行 = 原始 T-1 日的因子值
        factor = self._preprocess_factor(factor)

        # Step 1: 计算当日收益率
        # ret[T] = close[T] / close[T-1] - 1
        # 与 delay=1 后的 factor[T]（原 T-1 日因子）同行对齐：
        #   factor[T] 预测 ret[T]，即 T-1 日因子 → T 日收益，无未来函数
        ret = close / close.shift(1) - 1   # T×N，第 0 行全 NaN

        # Step 2: 构建原始权重矩阵（调仓日满足条件的股票平等或加权持仓）
        print("[VectorEngine] 构建持仓权重矩阵...")
        weights_raw = self._build_weights(factor, is_suspended, is_limit)

        # Step 3: 非调仓日沿用上一调仓日权重（前向填充）
        weights = weights_raw.ffill()
        weights = weights.fillna(0.0)

        # Step 4: 计算换手率
        turnover = Performance.calc_turnover(weights)

        # Step 5: 计算交易成本（换手率 × 单边成本率）
        cost_series = turnover * self.cost_rate

        # Step 6: 计算组合日收益率
        # weights[T] 由 factor[T]（经 delay=1 后即原 T-1 日因子）构建
        # ret[T] = close[T]/close[T-1] - 1
        # port_ret[T] = sum_i( weights[T,i] * ret[T,i] )
        # 两者同行直接相乘，无需额外 shift，语义清晰
        gross_ret = (weights * ret).sum(axis=1)
        gross_ret.iloc[0] = 0.0

        net_ret = gross_ret - cost_series
        net_ret = net_ret.fillna(0.0)
        gross_ret = gross_ret.fillna(0.0)

        # Step 7: 净值序列
        nav = (1 + net_ret).cumprod()
        nav.name = "nav"
        nav.iloc[0] = 1.0  # 确保从 1.0 开始

        # Step 8: IC 序列
        print("[VectorEngine] 计算 IC 序列...")
        # IC 计算：factor[T]（delay 后，即 T-1 日因子）vs ret[T]（T 日当日收益）
        # 同行对齐，直接计算截面 Rank IC
        ic_series = Performance.calc_ic_series(factor, ret)

        # Step 9: 汇总指标
        print("[VectorEngine] 汇总绩效指标...")
        metrics = Performance.summary(
            nav             = nav,
            daily_returns   = net_ret,
            weights         = weights,
            factor          = factor,
            forward_returns = ret,
            cost_series     = cost_series,
        )

        # Step 10: 获取实际调仓日期
        rebalance_dates = self._get_rebalance_dates(factor.index)

        print("[VectorEngine] 回测完成！\n")

        return BacktestResult(
            nav             = nav,
            daily_returns   = net_ret,
            gross_returns   = gross_ret,
            cost_series     = cost_series,
            weights         = weights,
            turnover        = turnover,
            ic_series       = ic_series,
            forward_returns = ret,
            factor          = factor,
            metrics         = metrics,
            rebalance_dates = rebalance_dates,
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _preprocess_factor(self, factor: pd.DataFrame) -> pd.DataFrame:
        """
        按顺序执行三步因子预处理（每步均可通过参数跳过）：

        1. delay  ：Ts_Delay(factor, self.delay)
                    delay=1 → factor 第 T 行变为原始 T-1 日的因子值
                    使得 factor[T] 与 ret[T] 对齐时，语义为"T-1日因子预测T日收益"
        2. decay  ：Decay_Linear(factor, self.decay) — 线性衰减，decay=0 跳过
        3. 中性化  ：Neutralize(factor, self.industry) — 行业中性化，industry=None 跳过

        Parameters
        ----------
        factor : 已对齐的因子矩阵 (T × N)

        Returns
        -------
        pd.DataFrame : 预处理后的因子矩阵，形状不变
        """
        from quant_alpha_engine.ops.alpha_ops import AlphaOps

        # Step 1: 延迟（delay=1：T行因子值 ← 原T-1日的值，与ret[T]对齐后无未来函数）
        if self.delay > 0:
            factor = AlphaOps.Ts_Delay(factor, self.delay)

        # Step 2: 线性衰减（decay=0 跳过）
        if self.decay > 0:
            factor = AlphaOps.Decay_Linear(factor, self.decay)

        # Step 3: 行业中性化（industry=None 跳过）
        if self.industry is not None:
            factor = AlphaOps.Neutralize(factor, self.industry)

        return factor

    def _align_data(self):
        """
        对齐所有输入数据，取共同日期和股票子集。
        同时对 is_suspended 和 is_limit 做类型转换。

        若设置了 start_date / end_date，则在取交集之后进一步裁剪日期范围。
        裁剪仅影响回测评估区间，不影响因子本身的历史窗口计算。
        """
        # 取共同日期
        common_dates = (
            self.factor.index
            .intersection(self.close.index)
            .intersection(self.is_suspended.index)
            .intersection(self.is_limit.index)
        )
        # 取共同股票
        common_stocks = (
            self.factor.columns
            .intersection(self.close.columns)
            .intersection(self.is_suspended.columns)
            .intersection(self.is_limit.columns)
        )

        # ── 按 start_date / end_date 裁剪回测评估区间 ─────────────────
        if self.start_date is not None:
            common_dates = common_dates[common_dates >= pd.Timestamp(self.start_date)]
        if self.end_date is not None:
            common_dates = common_dates[common_dates <= pd.Timestamp(self.end_date)]

        if len(common_dates) == 0:
            raise ValueError(
                f"裁剪后日期为空！请检查 start_date='{self.start_date}'、"
                f"end_date='{self.end_date}' 是否在数据范围内。"
            )

        factor       = self.factor.loc[common_dates, common_stocks]
        close        = self.close.loc[common_dates, common_stocks]
        is_suspended = self.is_suspended.loc[common_dates, common_stocks].fillna(False).astype(bool)
        is_limit     = self.is_limit.loc[common_dates, common_stocks].fillna(False).astype(bool)

        return factor, close, is_suspended, is_limit

    def _build_weights(
        self,
        factor:       pd.DataFrame,
        is_suspended: pd.DataFrame,
        is_limit:     pd.DataFrame,
    ) -> pd.DataFrame:
        """
        在调仓日根据因子值构建持仓权重矩阵。
        非调仓日权重为 NaN（后续通过 ffill 填充）。

        流程：
        1. 过滤停牌/涨跌停（因子值置 NaN）
        2. 按因子值排名取 Top-N
        3. 按选定方式分配权重
        4. 归一化（权重之和为 1）
        """
        n_dates, n_stocks = factor.shape
        weights = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)

        # 生成调仓日索引
        rebal_indices = self._get_rebalance_indices(n_dates)

        # 过滤后的因子（停牌/涨跌停置 NaN）
        tradable_mask  = ~(is_suspended | is_limit)
        factor_filtered = factor.where(tradable_mask, np.nan)

        top_n = min(self.top_n, n_stocks)

        for idx in rebal_indices:
            daily_factor = factor_filtered.iloc[idx]  # Series (N,)

            # 有效因子值
            valid = daily_factor.dropna()
            if len(valid) == 0:
                weights.iloc[idx] = 0.0
                continue

            # 取 Top-N（按因子值大小，降序）
            top_n_actual = min(top_n, len(valid))
            top_stocks = valid.nlargest(top_n_actual).index

            row_weights = pd.Series(0.0, index=factor.columns)

            if self.weight_method == "equal":
                row_weights[top_stocks] = 1.0 / top_n_actual

            elif self.weight_method == "factor_weighted":
                # 用因子值绝对值加权（取 Top-N 后归一化）
                raw_w = valid[top_stocks].abs()
                total = raw_w.sum()
                if total > 1e-10:
                    row_weights[top_stocks] = raw_w / total
                else:
                    row_weights[top_stocks] = 1.0 / top_n_actual

            else:
                raise ValueError(
                    f"weight_method 必须为 'equal' 或 'factor_weighted'，"
                    f"收到: '{self.weight_method}'"
                )

            weights.iloc[idx] = row_weights.values

        return weights

    def _get_rebalance_indices(self, n_dates: int) -> list:
        """返回调仓日的行索引列表。"""
        return list(range(0, n_dates, self.rebalance_freq))

    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex) -> list:
        """返回调仓日的日期列表。"""
        indices = self._get_rebalance_indices(len(date_index))
        return [date_index[i] for i in indices]


# ===========================================================================
# 控制台打印
# ===========================================================================

def _print_metrics_table(metrics: dict) -> None:
    """以 pandas Styler 表格形式输出绩效报告（Jupyter 显示为 HTML，终端回退为纯文本）。"""

    # ── 多档位阈值定义 ──────────────────────────────────────
    # 格式：(上界, 评价文字, 背景色HEX)，按上界从小到大排列，最后一条上界为 None
    _BANDS: dict[str, dict] = {
        "年化收益率": {
            "lower_is_better": False,
            "bands": [
                (0.15, "较差",   "#c0392b"),
                (0.25, "良好",   "#2980b9"),
                (None, "优秀",   "#27ae60"),
            ],
        },
        "Sharpe_Ratio": {
            "lower_is_better": False,
            "bands": [
                (1.0,  "较差",   "#c0392b"),
                (1.5,  "良好",   "#2980b9"),
                (None, "优秀",   "#27ae60"),
            ],
        },
        "最大回撤": {
            "lower_is_better": True,
            "bands": [
                (0.05, "优秀",   "#27ae60"),
                (0.10, "良好",   "#2980b9"),
                (None, "较差",   "#c0392b"),
            ],
        },
        "IC_Mean": {
            "lower_is_better": False,
            "bands": [
                (0.03, "较差",   "#c0392b"),
                (0.05, "良好",   "#2980b9"),
                (None, "优秀",   "#27ae60"),
            ],
        },
        "ICIR": {
            "lower_is_better": False,
            "bands": [
                (1.5,  "较差",   "#c0392b"),
                (2.5,  "良好",   "#2980b9"),
                (None, "优秀",   "#27ae60"),
            ],
        },
        "IC_胜率": {
            "lower_is_better": False,
            "bands": [
                (0.55, "较差",   "#c0392b"),
                (0.60, "良好",   "#2980b9"),
                (None, "优秀",   "#27ae60"),
            ],
        },
        "年化波动率": {
            "lower_is_better": True,
            "bands": [
                (0.10, "低风险",   "#27ae60"),
                (0.20, "中风险",   "#2980b9"),
                (0.30, "高风险",   "#c0392b"),
                (None, "极高风险", "#8e44ad"),
            ],
        },
        "Calmar_Ratio": {
            "lower_is_better": False,
            "bands": [
                (0.5,  "不合格", "#c0392b"),
                (1.0,  "合格",   "#d4ac0d"),
                (2.0,  "良好",   "#2980b9"),
                (None, "优秀",   "#27ae60"),
            ],
        },
        "IC_Std": {
            "lower_is_better": True,
            "bands": [
                (0.10, "极稳定", "#27ae60"),
                (0.15, "稳定",   "#2980b9"),
                (0.20, "波动",   "#d4ac0d"),
                (None, "不稳",   "#8e44ad"),
            ],
        },
    }

    def _fmt(key: str, val) -> str:
        """格式化指标值为字符串。"""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        pct_keys = {"年化收益率", "年化波动率", "最大回撤", "日均换手率", "年化手续费", "IC_胜率"}
        if key in pct_keys:
            return f"{val * 100:+.2f}%"
        elif key in {"Sharpe_Ratio", "Calmar_Ratio", "Fitness"}:
            return f"{val:+.4f}"
        else:
            return f"{val:+.4f}"

    def _evaluate(key: str, val) -> tuple[str, str]:
        """返回 (评价文字, 背景色HEX)；无定义则返回空字符串和空色。"""
        if key not in _BANDS:
            return "", ""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—", "#555555"
        cfg   = _BANDS[key]
        v     = abs(val)
        bands = cfg["bands"]
        if cfg["lower_is_better"]:
            for upper, label, color in bands:
                if upper is None or v <= upper:
                    return label, color
        else:
            for i in range(len(bands) - 1, -1, -1):
                lower = bands[i - 1][0] if i > 0 else 0.0
                if v >= lower:
                    return bands[i][1], bands[i][2]
        return bands[-1][1], bands[-1][2]

    # ── 组装表格数据 ─────────────────────────────────────────
    display_order = [
        ("年化收益率",  "年化收益率"),
        ("年化波动率",  "年化波动率"),
        ("Sharpe_Ratio", "Sharpe Ratio"),
        ("Calmar_Ratio", "Calmar Ratio"),
        ("最大回撤",    "最大回撤"),
        ("IC_Mean",     "IC 均值"),
        ("IC_Std",      "IC 标准差"),
        ("ICIR",        "ICIR"),
        ("IC_胜率",      "IC 胜率"),
        ("IC_t统计量",   "IC t-stat"),
        ("日均换手率",   "日均换手率"),
        ("年化手续费",   "年化手续费"),
        ("Fitness",     "Fitness"),
    ]

    rows       = []   # [(label, val_str, eval_text)]
    eval_colors = []  # [bg_hex_or_empty]

    for key, label in display_order:
        val  = metrics.get(key, np.nan)
        v_str = _fmt(key, val)
        e_text, e_color = _evaluate(key, val)
        rows.append((label, v_str, e_text))
        eval_colors.append(e_color)

    # ── 尝试用 pandas Styler 在 Jupyter 中渲染 HTML 表格 ────
    try:
        import pandas as _pd
        from IPython.display import display as _display

        df = _pd.DataFrame(rows, columns=["指标", "数值", "评价"])

        def _style_eval(col):
            styles = []
            for i, (_, _, e_text) in enumerate(rows):
                bg = eval_colors[i]
                if bg:
                    styles.append(
                        f"background-color: {bg}; color: white; "
                        f"font-weight: bold; text-align: center;"
                    )
                else:
                    styles.append("text-align: center; color: #888888;")
            return styles

        styler = (
            df.style
            .set_caption("QuantAlpha Engine — 回测绩效报告")
            .apply(_style_eval, subset=["评价"])
            .set_properties(subset=["指标"], **{"text-align": "left",  "font-weight": "bold"})
            .set_properties(subset=["数值"], **{"text-align": "right", "font-family": "monospace"})
            .set_table_styles([
                {"selector": "caption",
                 "props": "font-size:15px; font-weight:bold; padding:8px 0;"},
                {"selector": "th",
                 "props": "background-color:#2c3e50; color:white; "
                          "text-align:center; padding:6px 14px;"},
                {"selector": "td",
                 "props": "padding:5px 14px; border-bottom:1px solid #ddd;"},
                {"selector": "tr:hover td",
                 "props": "background-color:#f0f4f8;"},
            ])
            .hide(axis="index")
        )
        _display(styler)

    except Exception:
        # ── 终端回退：纯文本对齐输出 ────────────────────────
        print()
        print(f"{'指标':<14}  {'数值':>10}  {'评价'}")
        print("─" * 36)
        for (label, v_str, e_text), _ in zip(rows, eval_colors):
            print(f"{label:<14}  {v_str:>10}  {e_text}")
        print()
