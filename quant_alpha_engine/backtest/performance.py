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
    def calc_max_drawdown_detailed(nav: pd.Series) -> Dict[str, Any]:
        """
        最大回撤详情：值、起止日期、持续天数、是否已修复。

        定义：
            MaxDD_StartDate     ：进入最大回撤前的最后一个高点日期（峰值日）
            MaxDD_EndDate       ：净值跌到最低的那一天（谷底日）
            MaxDD_RecoveryDate  ：净值首次回到峰值的日期；若区间内未修复则为 None
            MaxDD_Duration      ：从峰值日到修复日（或区间末日）的交易日数

        Returns
        -------
        dict : keys = {MaxDD, MaxDD_Duration, MaxDD_StartDate,
                       MaxDD_EndDate, MaxDD_RecoveryDate, MaxDD_Recovered}
        """
        empty = {
            "MaxDD":              0.0,
            "MaxDD_Duration":     0,
            "MaxDD_StartDate":    None,
            "MaxDD_EndDate":      None,
            "MaxDD_RecoveryDate": None,
            "MaxDD_Recovered":    True,
        }
        if nav is None or len(nav) < 2:
            return empty

        rolling_max = nav.cummax()
        drawdown    = (nav - rolling_max) / rolling_max
        end_idx     = drawdown.idxmin()
        mdd_val     = float(drawdown.loc[end_idx])
        if mdd_val == 0.0:
            return empty

        peak_val = float(rolling_max.loc[end_idx])
        # 峰值日：end 之前最后一个 nav >= peak 的日期
        pre = nav.loc[:end_idx]
        start_idx = pre[pre >= peak_val].index[-1] if (pre >= peak_val).any() else nav.index[0]
        # 修复日：end 之后首次 nav >= peak
        post = nav.loc[end_idx:]
        rec_mask = post >= peak_val
        recovery_idx = post[rec_mask].index[0] if rec_mask.any() else None
        recovered = recovery_idx is not None
        # 持续天数：从峰值到（修复日 or 区间末日）
        end_for_dur = recovery_idx if recovered else nav.index[-1]
        duration = int(nav.loc[start_idx:end_for_dur].shape[0])

        return {
            "MaxDD":              mdd_val,
            "MaxDD_Duration":     duration,
            "MaxDD_StartDate":    start_idx,
            "MaxDD_EndDate":      end_idx,
            "MaxDD_RecoveryDate": recovery_idx,
            "MaxDD_Recovered":    recovered,
        }

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
    # 分组（Quantile）单调性
    # ==================================================================

    @staticmethod
    def calc_quantile_returns(
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
        n_quantiles: int = 10,
    ) -> pd.DataFrame:
        """
        计算 N 分组的每日等权组合收益率。

        每个截面按 factor 值升序分成 n_quantiles 组（Q1=最小，Qn=最大），
        每组内部等权持有，输出 T × n_quantiles 的日收益率矩阵。

        Parameters
        ----------
        factor          : T × N 因子矩阵（已 delay 等预处理）
        forward_returns : T × N 同行对齐的收益矩阵 (ret[T] = close[T]/close[T-1]-1)
        n_quantiles     : 分组数，默认 10

        Returns
        -------
        pd.DataFrame : index=日期, columns=['Q1', 'Q2', ..., 'Qn']
        """
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles 必须 >= 2，收到 {n_quantiles}")

        factor, forward_returns = factor.align(forward_returns, join="inner")
        # 截面分组（按因子值百分位）；NaN 自动跳过
        # qcut 在每行做，duplicates='drop' 避免相等值过多导致的边界冲突
        ranks = factor.rank(axis=1, pct=True)  # 0~1
        # 等价于按百分位切桶，避免逐行 qcut 的开销
        bins = np.floor(ranks * n_quantiles).clip(upper=n_quantiles - 1)

        cols = [f"Q{i + 1}" for i in range(n_quantiles)]
        out  = pd.DataFrame(np.nan, index=factor.index, columns=cols)

        ret_arr = forward_returns.values
        bin_arr = bins.values
        for q in range(n_quantiles):
            mask = (bin_arr == q)
            counts = mask.sum(axis=1)
            # 等权：组内平均
            with np.errstate(invalid="ignore", divide="ignore"):
                masked_ret = np.where(mask, ret_arr, 0.0)
                row_sum    = np.nansum(masked_ret, axis=1)
                out.iloc[:, q] = np.where(counts > 0, row_sum / counts, np.nan)
        return out

    @staticmethod
    def calc_monotonicity_score(quantile_returns: pd.DataFrame) -> float:
        """
        分组单调性得分：分组累计收益（或均值）与组号的 Spearman 秩相关。

        值域 [-1, 1]：+1 表示 Q1<Q2<...<Qn 完美单调，-1 表示完美反向，
        0 附近表示无单调性（因子无效）。

        Parameters
        ----------
        quantile_returns : calc_quantile_returns 输出（T × n_quantiles）

        Returns
        -------
        float
        """
        if quantile_returns is None or quantile_returns.empty:
            return float("nan")
        # 用每组的日收益均值（≈ 年化收益的代理）做 rank corr
        mean_ret = quantile_returns.mean(axis=0).values
        if np.all(np.isnan(mean_ret)) or len(mean_ret) < 2:
            return float("nan")
        q_idx = np.arange(1, len(mean_ret) + 1, dtype=float)
        # Spearman = Pearson on ranks
        r1 = pd.Series(mean_ret).rank().values
        r2 = pd.Series(q_idx).rank().values
        if np.std(r1) < 1e-12 or np.std(r2) < 1e-12:
            return 0.0
        return float(np.corrcoef(r1, r2)[0, 1])

    @staticmethod
    def calc_long_short_sharpe(
        quantile_returns: pd.DataFrame,
        cost_rate: float = 0.0,
    ) -> Dict[str, float]:
        """
        多空头组合（Top - Bottom）夏普与年化收益。

        多头 = 最高分组 (Qn)，空头 = 最低分组 (Q1)；ls_ret = Qn - Q1。
        cost_rate 仅作为粗略扣费（多空两侧 100% 换仓的悲观估计：2 * cost_rate）。

        Parameters
        ----------
        quantile_returns : T × n_quantiles 矩阵
        cost_rate        : 单边成本，默认 0（不扣费）

        Returns
        -------
        dict : {'LS_Sharpe', 'LS_AnnReturn', 'LS_AnnVol', 'LS_MaxDD'}
        """
        if quantile_returns is None or quantile_returns.shape[1] < 2:
            return {"LS_Sharpe": np.nan, "LS_AnnReturn": np.nan,
                    "LS_AnnVol": np.nan, "LS_MaxDD": np.nan}
        long_ret  = quantile_returns.iloc[:, -1]
        short_ret = quantile_returns.iloc[:, 0]
        ls_ret    = (long_ret - short_ret).fillna(0.0) - 2 * cost_rate
        ls_nav    = (1 + ls_ret).cumprod()
        return {
            "LS_Sharpe":    Performance.calc_sharpe(ls_ret),
            "LS_AnnReturn": Performance.calc_annualized_return(ls_nav),
            "LS_AnnVol":    Performance.calc_annualized_volatility(ls_ret),
            "LS_MaxDD":     Performance.calc_max_drawdown(ls_nav),
        }

    # ==================================================================
    # 多窗口 IC / IC 衰减
    # ==================================================================

    @staticmethod
    def calc_ic_decay(
        factor: pd.DataFrame,
        close: pd.DataFrame,
        horizons=(1, 5, 10, 20),
    ) -> pd.DataFrame:
        """
        多窗口 IC 衰减曲线：对每个 horizon h，
        计算 factor[T] 与 (close[T+h]/close[T] - 1) 的截面 Rank IC 时序。

        注意：传入的 factor 应已经过 delay 处理（与 close 同行对齐时无未来函数）。
        然后对每个 h 计算 forward h 日收益（close 自身向前 shift），与 factor 对齐相关。

        Parameters
        ----------
        factor   : T × N 因子矩阵（建议传入已 delay 后的 factor）
        close    : T × N 收盘价矩阵
        horizons : 前向窗口列表，默认 (1, 5, 10, 20)

        Returns
        -------
        pd.DataFrame : index=日期, columns=['IC_h1','IC_h5','IC_h10','IC_h20']
        """
        factor, close = factor.align(close, join="inner")
        out = {}
        for h in horizons:
            # 前向 h 日收益：fwd_ret[T] = close[T+h]/close[T] - 1
            fwd_ret = close.shift(-h) / close - 1
            ic = Performance.calc_ic_series(factor, fwd_ret)
            out[f"IC_h{h}"] = ic
        return pd.DataFrame(out)

    @staticmethod
    def calc_ic_decay_summary(ic_decay: pd.DataFrame) -> Dict[str, float]:
        """
        把 IC 衰减矩阵汇总成 {horizon: IC_Mean} 字典。
        """
        if ic_decay is None or ic_decay.empty:
            return {}
        return {col: float(ic_decay[col].mean()) for col in ic_decay.columns}

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
        mdd_info = Performance.calc_max_drawdown_detailed(nav)
        mdd     = mdd_info["MaxDD"]
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
            "最大回撤_天数":   mdd_info["MaxDD_Duration"],
            "最大回撤_峰值日": mdd_info["MaxDD_StartDate"],
            "最大回撤_谷底日": mdd_info["MaxDD_EndDate"],
            "最大回撤_修复日": mdd_info["MaxDD_RecoveryDate"],
            "最大回撤_已修复": mdd_info["MaxDD_Recovered"],
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
