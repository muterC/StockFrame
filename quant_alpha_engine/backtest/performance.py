"""
Performance — 绩效指标计算模块
================================
计算 QuantAlpha_Engine 所有核心回测指标：
  - Sharpe Ratio（年化）
  - Maximum Drawdown（最大回撤）
  - IC / Rank IC（截面信息系数）
  - ICIR（IC 信息比率）
  - Turnover（日均换手率）
  - Fitness（世坤核心综合指标）
  - 年化收益率、年化波动率
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class Performance:
    """
    绩效指标计算工具类（全静态方法）。

    所有方法均为纯函数，无状态，可独立调用。
    """

    ANNUALIZE_FACTOR = 252  # 年化因子（交易日数）

    # ==================================================================
    # 基础收益指标
    # ==================================================================

    @staticmethod
    def calc_annualized_return(nav: pd.Series) -> float:
        """
        计算年化收益率。

        公式：(nav_T / nav_0)^(252/T) - 1

        Parameters
        ----------
        nav : 净值序列（从 1.0 开始）

        Returns
        -------
        float : 年化收益率，如 0.15 表示 15%
        """
        if len(nav) < 2:
            return 0.0
        n_days = len(nav)
        total_return = nav.iloc[-1] / nav.iloc[0]
        if total_return <= 0:
            return -1.0
        ann_return = total_return ** (Performance.ANNUALIZE_FACTOR / n_days) - 1
        return float(ann_return)

    @staticmethod
    def calc_annualized_volatility(returns: pd.Series) -> float:
        """
        计算年化波动率。

        Parameters
        ----------
        returns : 日收益率序列

        Returns
        -------
        float : 年化波动率
        """
        clean = returns.dropna()
        if len(clean) < 2:
            return 0.0
        return float(clean.std(ddof=1) * np.sqrt(Performance.ANNUALIZE_FACTOR))

    @staticmethod
    def calc_sharpe(
        returns: pd.Series,
        risk_free: float = 0.0,
        annualize: bool = True,
    ) -> float:
        """
        计算夏普比率（Sharpe Ratio）。

        公式：(mean(r) - rf) / std(r) * sqrt(252)

        Parameters
        ----------
        returns    : 日收益率序列
        risk_free  : 无风险利率（年化），默认 0
        annualize  : 是否年化，默认 True

        Returns
        -------
        float
        """
        clean = returns.dropna()
        if len(clean) < 2 or clean.std() < 1e-10:
            return 0.0
        daily_rf = risk_free / Performance.ANNUALIZE_FACTOR
        excess   = clean - daily_rf
        sharpe   = excess.mean() / excess.std(ddof=1)
        if annualize:
            sharpe *= np.sqrt(Performance.ANNUALIZE_FACTOR)
        return float(sharpe)

    @staticmethod
    def calc_max_drawdown(nav: pd.Series) -> float:
        """
        计算最大回撤（Maximum Drawdown）。

        公式：min((nav_t - max(nav_{0..t})) / max(nav_{0..t}))

        Parameters
        ----------
        nav : 净值序列

        Returns
        -------
        float : 最大回撤（负数），如 -0.15 表示 -15%
        """
        if len(nav) < 2:
            return 0.0
        rolling_max = nav.cummax()
        drawdown    = (nav - rolling_max) / rolling_max
        return float(drawdown.min())

    @staticmethod
    def calc_calmar(nav: pd.Series) -> float:
        """
        Calmar 比率 = 年化收益率 / |最大回撤|。
        """
        ann_ret = Performance.calc_annualized_return(nav)
        mdd     = Performance.calc_max_drawdown(nav)
        if abs(mdd) < 1e-10:
            return 0.0
        return float(ann_ret / abs(mdd))

    # ==================================================================
    # IC 信息系数
    # ==================================================================

    @staticmethod
    def calc_ic_series(
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> pd.Series:
        """
        计算每日截面 Rank IC（Spearman 秩相关系数）。

        实现方式：
        1. 对每日截面，将因子值和前向收益分别做百分比排名
        2. 计算两者的 Pearson 相关系数（等价于 Spearman 相关系数）
        使用向量化矩阵运算，避免逐行循环。

        Parameters
        ----------
        factor          : 因子矩阵 (T × N)
        forward_returns : 对应的前向 N 日收益率矩阵 (T × N)

        Returns
        -------
        pd.Series : 每日 IC 值，index 为日期
        """
        # 对齐数据
        factor, forward_returns = factor.align(forward_returns, join="inner")

        # 截面百分比排名（axis=1，忽略 NaN）
        rf = factor.rank(axis=1, pct=True)          # rank of factor
        rr = forward_returns.rank(axis=1, pct=True)  # rank of returns

        # 向量化 Pearson 相关（≡ Spearman）
        # IC_t = Corr(rf_t, rr_t) across stocks
        rf_demean = rf.sub(rf.mean(axis=1), axis=0)
        rr_demean = rr.sub(rr.mean(axis=1), axis=0)

        numerator   = (rf_demean * rr_demean).sum(axis=1)
        denom_rf    = np.sqrt((rf_demean ** 2).sum(axis=1))
        denom_rr    = np.sqrt((rr_demean ** 2).sum(axis=1))
        denominator = denom_rf * denom_rr

        ic = numerator / denominator.replace(0, np.nan)
        ic.name = "IC"
        return ic

    @staticmethod
    def calc_ic_stats(ic_series: pd.Series) -> Dict[str, float]:
        """
        计算 IC 统计摘要。

        Parameters
        ----------
        ic_series : 每日 IC 序列

        Returns
        -------
        dict with keys:
            IC_Mean          : IC 均值
            IC_Std           : IC 标准差
            ICIR             : IC 信息比率 (IC_Mean / IC_Std * sqrt(252))
            IC_Positive_Ratio: IC 为正的比例（胜率）
            IC_t_stat        : IC 均值的 t 统计量
        """
        clean = ic_series.dropna()
        if len(clean) == 0:
            return {k: np.nan for k in ["IC_Mean", "IC_Std", "ICIR",
                                         "IC_Positive_Ratio", "IC_t_stat"]}
        mean_ = clean.mean()
        std_  = clean.std(ddof=1)

        icir = (mean_ / std_ * np.sqrt(Performance.ANNUALIZE_FACTOR)) if std_ > 1e-10 else 0.0
        t_stat = mean_ / (std_ / np.sqrt(len(clean))) if std_ > 1e-10 else 0.0

        return {
            "IC_Mean":           float(mean_),
            "IC_Std":            float(std_),
            "ICIR":              float(icir),
            "IC_Positive_Ratio": float((clean > 0).mean()),
            "IC_t_stat":         float(t_stat),
        }

    # ==================================================================
    # 换手率
    # ==================================================================

    @staticmethod
    def calc_turnover(weights: pd.DataFrame) -> pd.Series:
        """
        计算每日单边换手率。

        换手率 = sum(|w_t - w_{t-1}|) / 2

        Parameters
        ----------
        weights : 持仓权重矩阵 (T × N)

        Returns
        -------
        pd.Series : 每日换手率
        """
        diff   = weights.diff().abs()
        to     = diff.sum(axis=1) / 2
        to.name = "turnover"
        return to

    # ==================================================================
    # Fitness（世坤核心指标）
    # ==================================================================

    @staticmethod
    def calc_fitness(
        sharpe: float,
        nav: pd.Series,
        turnover: pd.Series,
    ) -> float:
        """
        计算 Fitness 指标。

        公式：Fitness = Sharpe × sqrt(|年化收益率| / 平均换手率)

        WorldQuant 用于综合评估因子的收益质量与稳定性。

        Parameters
        ----------
        sharpe   : 年化夏普比率
        nav      : 净值序列（用于计算年化收益率）
        turnover : 每日换手率序列

        Returns
        -------
        float
        """
        ann_ret  = abs(Performance.calc_annualized_return(nav))
        avg_to   = turnover.dropna().mean()

        if avg_to < 1e-10 or ann_ret < 1e-10:
            return 0.0

        fitness = sharpe * np.sqrt(ann_ret / avg_to)
        return float(fitness)

    # ==================================================================
    # 汇总
    # ==================================================================

    @staticmethod
    def summary(
        nav: pd.Series,
        daily_returns: pd.Series,
        weights: pd.DataFrame,
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
        cost_series: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        计算所有核心绩效指标并汇总为字典。

        Parameters
        ----------
        nav             : 净值序列
        daily_returns   : 日净收益率（扣除成本后）
        weights         : 持仓权重矩阵
        factor          : 因子矩阵
        forward_returns : 前向收益矩阵
        cost_series     : 每日交易成本序列（可选）

        Returns
        -------
        dict : 包含所有核心指标
        """
        sharpe  = Performance.calc_sharpe(daily_returns)
        ann_ret = Performance.calc_annualized_return(nav)
        ann_vol = Performance.calc_annualized_volatility(daily_returns)
        mdd     = Performance.calc_max_drawdown(nav)
        calmar  = Performance.calc_calmar(nav)

        turnover  = Performance.calc_turnover(weights)
        ic_series = Performance.calc_ic_series(factor, forward_returns)
        ic_stats  = Performance.calc_ic_stats(ic_series)
        fitness   = Performance.calc_fitness(sharpe, nav, turnover)

        ann_cost = (
            cost_series.dropna().mean() * Performance.ANNUALIZE_FACTOR
            if cost_series is not None and len(cost_series.dropna()) > 0
            else np.nan
        )

        metrics = {
            # 收益指标
            "年化收益率":     ann_ret,
            "年化波动率":     ann_vol,
            "Sharpe_Ratio":  sharpe,
            "Calmar_Ratio":  calmar,
            "最大回撤":       mdd,
            # IC 指标
            "IC_Mean":       ic_stats["IC_Mean"],
            "IC_Std":        ic_stats["IC_Std"],
            "ICIR":          ic_stats["ICIR"],
            "IC_胜率":        ic_stats["IC_Positive_Ratio"],
            "IC_t统计量":     ic_stats["IC_t_stat"],
            # 换手与成本
            "日均换手率":     float(turnover.dropna().mean()),
            "年化手续费":     ann_cost,
            # 综合指标
            "Fitness":       fitness,
        }
        return metrics
