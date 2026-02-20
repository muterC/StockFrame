"""
VectorEngine — 矩阵式净值回测引擎
====================================
基于因子权重矩阵的向量化回测，严格防止未来函数（Look-ahead bias）。

回测逻辑
--------
1. 数据对齐（factor / close / is_suspended / is_limit 取交集）
2. 计算前向 1 日收益率（T 日因子 → T+1 日收益）
3. 按调仓频率生成持仓权重矩阵（Top-N 等权 or 因子加权）
4. 过滤停牌/涨跌停股票（权重置零并重新归一化）
5. 扣除单边交易成本（手续费 + 滑点）
6. 输出净值序列及全量绩效指标

防止未来函数原则
--------------
- 权重使用当日收盘后的因子值生成
- 收益用 close.shift(-1)/close - 1（即明日收益）
- 持仓收益计算时用 weights.shift(1)（昨日权重 × 今日收益）
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal

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

    Examples
    --------
    >>> engine = VectorEngine(
    ...     factor=factor_df,
    ...     close=close_df,
    ...     is_suspended=suspended_df,
    ...     is_limit=limit_df,
    ...     rebalance_freq=5,
    ...     top_n=30,
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

        print(f"[VectorEngine] 数据规模：{len(factor)} 个交易日 × {factor.shape[1]} 只股票")

        # Step 1: 计算前向 1 日收益率（T 日价格 → T+1 日实际收益）
        fwd_ret = (close.shift(-1) / close - 1)   # T×N，最后一行全 NaN

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
        # 严格防未来函数：今日持仓（weights）已知，明日收益（fwd_ret.shift(1)）
        # 等价写法：昨日建仓的权重乘以今日实际收益
        # port_ret_t = sum_i(w_{t-1,i} * r_{t,i})
        # 其中 r_{t,i} = close_{t,i}/close_{t-1,i} - 1 = fwd_ret_{t-1,i}
        # 所以：port_ret_t = (weights.shift(1) * fwd_ret.shift(1)).sum(axis=1)
        # 但习惯上用 fwd_ret 直接表示：
        # w_t 在 t 日收盘后建仓，t+1 日获得 fwd_ret_t 的收益
        # port_ret_{t+1} = (w_t * fwd_ret_t).sum()
        # 即：port_gross = (weights * fwd_ret).sum(axis=1).shift(1)
        gross_ret = (weights * fwd_ret).sum(axis=1).shift(1)
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
        # IC 计算：当日因子 vs 当日前向收益（对齐到同一 index）
        ic_series = Performance.calc_ic_series(factor, fwd_ret)

        # Step 9: 汇总指标
        print("[VectorEngine] 汇总绩效指标...")
        metrics = Performance.summary(
            nav             = nav,
            daily_returns   = net_ret,
            weights         = weights,
            factor          = factor,
            forward_returns = fwd_ret,
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
            forward_returns = fwd_ret,
            factor          = factor,
            metrics         = metrics,
            rebalance_dates = rebalance_dates,
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _align_data(self):
        """
        对齐所有输入数据，取共同日期和股票子集。
        同时对 is_suspended 和 is_limit 做类型转换。
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
    """使用 Unicode 制表符在控制台打印美观的绩效指标表格。"""

    def fmt(key: str, val) -> str:
        """格式化指标值。"""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "  N/A"
        pct_keys = {"年化收益率", "年化波动率", "最大回撤", "日均换手率", "年化手续费", "IC_胜率"}
        if key in pct_keys:
            return f"{val * 100:>8.2f}%"
        elif key in {"IC_Mean", "IC_Std", "ICIR", "IC_t统计量"}:
            return f"{val:>8.4f} "
        else:
            return f"{val:>8.4f} "

    title = " QuantAlpha Engine — 回测绩效报告 "
    width = 46

    print("\n" + "╔" + "═" * width + "╗")
    print("║" + title.center(width) + "║")
    print("╠" + "═" * width + "╣")

    display_order = [
        ("年化收益率",  "年化收益率"),
        ("年化波动率",  "年化波动率"),
        ("Sharpe_Ratio", "Sharpe Ratio"),
        ("Calmar_Ratio",  "Calmar Ratio"),
        ("最大回撤",     "最大回撤"),
        ("IC_Mean",      "IC 均值"),
        ("IC_Std",       "IC 标准差"),
        ("ICIR",         "ICIR"),
        ("IC_胜率",       "IC 胜率"),
        ("IC_t统计量",    "IC t-stat"),
        ("日均换手率",    "日均换手率"),
        ("年化手续费",    "年化手续费"),
        ("Fitness",      "Fitness"),
    ]

    for key, label in display_order:
        val = metrics.get(key, np.nan)
        val_str = fmt(key, val)
        label_padded = label.ljust(16)
        print(f"║  {label_padded}│{val_str.rjust(26)}  ║")

    print("╚" + "═" * width + "╝\n")
