"""
AlphaOps — QuantAlpha_Engine 算子库
=====================================
所有算子均支持 pandas.DataFrame 向量化运算。
约定：DataFrame 的 Index 为时间，Columns 为股票代码。

算子分类
--------
时序类 (Time-Series)
    Ts_Sum, Ts_Mean, Ts_Max, Ts_Min, Ts_Delta, Ts_Delay,
    Ts_Std, Ts_Rank, Ts_Corr,
    Ts_Skew, Ts_Kurt, Ts_Autocorr, Ts_Hurst

截面类 (Cross-Sectional)
    Rank, ZScore, Scale

特殊类 (Special)
    Decay_Linear, Neutralize

量价因子 (Price-Volume)
    VWAP, VWAP_Bias, PVDeviation, Amihud

动量因子 (Momentum)
    RiskAdjMomentum, PricePathQuality, RangeBreakout

技术指标 (Technical)
    RSI, KDJ, MACD

缩量稳价因子 (VolSpike-PriceStable)
    VolSpike, PriceVarShrink, PriceMeanShrink, VolSpikeStablePrice

用法示例
--------
>>> from quant_alpha_engine.ops import AlphaOps as op
>>> factor = op.Rank(op.Ts_Delta(close, 20))
>>> factor2 = op.Neutralize(op.Rank(op.Ts_Corr(volume, close, 10)), industry)
>>> f_rsi  = op.RSI(close, window=14)
>>> f_macd = op.MACD(close, fast=12, slow=26, signal=9)
>>> # 缩量稳价复合因子（量能异动 + 价格收敛信号）
>>> f_vssp = op.VolSpikeStablePrice(turnover, close, open_, n_short=3, n_long=10)
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# 屏蔽 rolling apply 中可能出现的 NaN 警告
warnings.filterwarnings("ignore", category=RuntimeWarning)


class AlphaOps:
    """
    因子算子库（全静态方法）。

    所有输入 DataFrame 须满足：
        - Index: pd.DatetimeIndex（时间升序）
        - Columns: 股票代码

    NaN 值处理原则：滚动窗口不足时返回 NaN，截面操作自动忽略 NaN。
    """

    # ==================================================================
    # 时序类算子 (Time-Series)
    # ==================================================================

    @staticmethod
    def Ts_Sum(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """滑动窗口求和。

        Parameters
        ----------
        df     : 输入因子矩阵
        window : 窗口大小（天数）

        Returns
        -------
        pd.DataFrame
            与 df 同形状，前 window-1 行为 NaN
        """
        return df.rolling(window=window, min_periods=window).sum()

    @staticmethod
    def Ts_Mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """滑动窗口均值（移动平均）。"""
        return df.rolling(window=window, min_periods=window).mean()

    @staticmethod
    def Ts_Max(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """滑动窗口最大值。"""
        return df.rolling(window=window, min_periods=window).max()

    @staticmethod
    def Ts_Min(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """滑动窗口最小值。"""
        return df.rolling(window=window, min_periods=window).min()

    @staticmethod
    def Ts_Std(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """滑动窗口标准差。"""
        return df.rolling(window=window, min_periods=window).std()

    @staticmethod
    def Ts_Delta(df: pd.DataFrame, period: int) -> pd.DataFrame:
        """当前值与 period 天前的差值 (df - df.shift(period))。

        Parameters
        ----------
        period : 回看期数（天）
        """
        return df.diff(periods=period)

    @staticmethod
    def Ts_Delay(df: pd.DataFrame, period: int) -> pd.DataFrame:
        """数据滞后 period 天（等价于 df.shift(period)）。"""
        return df.shift(periods=period)

    @staticmethod
    def Ts_Rank(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        窗口内的时序百分比排名（当前值在过去 window 个观测中的分位数）。

        实现说明：逐列调用 rolling.apply + rankdata，最终值为当前时刻在
        窗口内的百分比排名（0~1）。
        注：因使用 apply，对大矩阵速度较慢；可考虑用 bottleneck 加速。

        Returns
        -------
        pd.DataFrame : 值域 [0, 1]
        """
        def _rank_last(x: np.ndarray) -> float:
            """返回数组最后一个元素的百分比排名。"""
            if np.all(np.isnan(x)):
                return np.nan
            valid = x[~np.isnan(x)]
            if len(valid) == 0:
                return np.nan
            # 当前值（最后一个非 NaN 值）在窗口内的百分比排名
            curr = x[-1]
            if np.isnan(curr):
                return np.nan
            rank = np.sum(valid <= curr) / len(valid)
            return rank

        return df.rolling(window=window, min_periods=window).apply(
            _rank_last, raw=True
        )

    @staticmethod
    def Ts_Corr(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """
        窗口内两个因子矩阵的滚动相关系数（逐列 Pearson 相关）。

        Parameters
        ----------
        df1, df2 : 形状相同的 DataFrame（时间 × 股票）
        window   : 滚动窗口大小

        Returns
        -------
        pd.DataFrame : 与输入同形状，值域 [-1, 1]
        """
        # 对齐两个 DataFrame
        df1, df2 = df1.align(df2, join="inner")

        result_dict = {}
        for col in df1.columns:
            s1 = df1[col]
            s2 = df2[col]
            result_dict[col] = s1.rolling(window=window, min_periods=window).corr(s2)

        return pd.DataFrame(result_dict, index=df1.index)[df1.columns]

    # ==================================================================
    # 截面类算子 (Cross-Sectional)
    # ==================================================================

    @staticmethod
    def Rank(df: pd.DataFrame) -> pd.DataFrame:
        """
        全市场截面百分比排名（0~1，值越大排名越靠前）。

        对每个时间截面，对所有股票的因子值做百分比排名（升序）。
        NaN 值不参与排名，对应位置仍为 NaN。

        Returns
        -------
        pd.DataFrame : 值域 [0, 1]
        """
        return df.rank(axis=1, pct=True, na_option="keep")

    @staticmethod
    def ZScore(df: pd.DataFrame) -> pd.DataFrame:
        """
        截面标准化（Z-Score Normalization）。

        每个时间截面减去截面均值后除以截面标准差。
        至少需要 2 个非 NaN 值，否则返回 NaN。

        Returns
        -------
        pd.DataFrame : 均值=0，标准差=1
        """
        mean_ = df.mean(axis=1)
        std_  = df.std(axis=1, ddof=1)
        # 避免除以零
        std_[std_ < 1e-10] = np.nan
        return df.sub(mean_, axis=0).div(std_, axis=0)

    @staticmethod
    def Scale(df: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
        """
        将截面绝对值之和缩放至 a。

        sum(|scaled_i|) = a（每个截面独立缩放）。
        用于将因子值缩放为名义持仓权重之和等于 a 的形式。

        Parameters
        ----------
        a : 目标绝对值之和，默认 1.0

        Returns
        -------
        pd.DataFrame
        """
        abs_sum = df.abs().sum(axis=1)
        abs_sum[abs_sum < 1e-10] = np.nan  # 避免除以零
        return df.div(abs_sum, axis=0).mul(a)

    # ==================================================================
    # 特殊类算子 (Special)
    # ==================================================================

    @staticmethod
    def Decay_Linear(df: pd.DataFrame, d: int) -> pd.DataFrame:
        """
        线性加权移动平均衰减（WorldQuant 核心算子）。

        权重向量：w = [1, 2, ..., d]（最近数据权重最大），归一化后求加权均值。
        即：decay(t) = sum_{k=0}^{d-1} w_{d-k} * x_{t-k} / sum(w)

        Parameters
        ----------
        d : 衰减窗口大小

        Returns
        -------
        pd.DataFrame : 线性衰减加权平均值
        """
        # 权重向量 [1, 2, ..., d]，最旧的权重为 1，最新的权重为 d
        weights = np.arange(1, d + 1, dtype=np.float64)
        weights /= weights.sum()

        def _weighted_mean(x: np.ndarray) -> float:
            if np.any(np.isnan(x)):
                # 有 NaN 时用有效数据重新加权
                valid_mask = ~np.isnan(x)
                if valid_mask.sum() < max(1, d // 2):
                    return np.nan
                w = weights[valid_mask]
                w = w / w.sum()
                return np.dot(w, x[valid_mask])
            return np.dot(weights, x)

        return df.rolling(window=d, min_periods=max(1, d // 2)).apply(
            _weighted_mean, raw=True
        )

    @staticmethod
    def Neutralize(
        df: pd.DataFrame,
        group_data: Union[pd.Series, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        行业中性化：利用 OLS 残差法剔除行业暴露。

        对每个时间截面，以行业虚拟变量为自变量做 OLS 回归，取残差作为
        中性化后的因子值。残差不含行业共性成分。

        Parameters
        ----------
        df         : 需要中性化的因子矩阵 (T × N)
        group_data : 行业映射
            - pd.Series: index=股票代码，values=行业名称（静态行业）
            - pd.DataFrame: (T × N)，支持动态行业变更（暂不实现，默认取最后一行）

        Returns
        -------
        pd.DataFrame : 行业中性化后的因子矩阵，均值接近0

        Notes
        -----
        实现细节：
        1. 构建行业哑变量矩阵 X (N × n_industries)
        2. 对当日有效股票子集，OLS: factor = X * beta + residual
        3. 取 residual 作为新的因子值
        """
        # 处理 group_data 格式
        if isinstance(group_data, pd.DataFrame):
            # 取最后一行作为静态行业分类
            group_series = group_data.iloc[-1]
        else:
            group_series = group_data  # pd.Series

        # 对齐股票列表
        common_stocks = df.columns.intersection(group_series.index)
        df_aligned = df[common_stocks]
        group_aligned = group_series[common_stocks]

        # 预构建哑变量矩阵（静态）
        dummies = pd.get_dummies(group_aligned, dtype=np.float64).values  # (N, K)

        result = pd.DataFrame(np.nan, index=df.index, columns=df.columns)

        for date in df.index:
            row = df_aligned.loc[date].values  # (N,)
            valid_mask = ~np.isnan(row)

            if valid_mask.sum() < dummies.shape[1] + 1:
                # 有效股票数量不足，无法回归
                result.loc[date, common_stocks] = row
                continue

            X = dummies[valid_mask]       # (n_valid, K)
            y = row[valid_mask]           # (n_valid,)

            # OLS: min ||y - X*beta||^2
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                residuals = y - X @ beta
                # 将残差写回
                res_full = np.full(valid_mask.shape, np.nan)
                res_full[valid_mask] = residuals
                result.loc[date, common_stocks] = res_full
            except np.linalg.LinAlgError:
                result.loc[date, common_stocks] = row

        return result

    # ==================================================================
    # 时序类扩展算子 (Time-Series Extended)
    # ==================================================================

    @staticmethod
    def Ts_Skew(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        滚动偏度（三阶标准矩）。

        衡量窗口内收益率分布的不对称程度：
          - 正偏度（右偏）：右尾长，极端正收益概率更高
          - 负偏度（左偏）：左尾长，极端负收益概率更高

        数学定义：
            Skew = E[(X - μ)³] / σ³

        Parameters
        ----------
        df     : 输入矩阵 (T × N)，通常为收益率序列
        window : 滚动窗口大小（至少需要 3 个样本）

        Returns
        -------
        pd.DataFrame : 与 df 同形状，前 window-1 行为 NaN
        """
        return df.rolling(window=window, min_periods=max(3, window)).skew()

    @staticmethod
    def Ts_Kurt(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        滚动峰度（四阶标准矩，超额峰度）。

        衡量窗口内分布尾部厚度：
          - 正超额峰度（> 0）：尖峰厚尾，极端值更频繁（正态分布 = 0）
          - 负超额峰度（< 0）：扁峰薄尾，分布更集中

        数学定义（Fisher 超额峰度）：
            Kurt = E[(X - μ)⁴] / σ⁴ - 3

        Parameters
        ----------
        df     : 输入矩阵 (T × N)
        window : 滚动窗口大小（至少需要 4 个样本）

        Returns
        -------
        pd.DataFrame : 超额峰度，正态分布对应值为 0
        """
        return df.rolling(window=window, min_periods=max(4, window)).kurt()

    @staticmethod
    def Ts_Autocorr(
        df: pd.DataFrame,
        lag: int,
        window: int,
    ) -> pd.DataFrame:
        """
        滚动自相关系数（lag 阶 Pearson 自相关）。

        衡量时间序列在 lag 期滞后下的线性相关性：
          - 正自相关：序列有趋势性（动量）
          - 负自相关：序列有反转性（均值回归）

        数学定义：
            AutoCorr(lag) = corr(Xₜ, Xₜ₋ₗₐ₉) in rolling window

        Parameters
        ----------
        df     : 输入矩阵 (T × N)
        lag    : 滞后阶数（天数）
        window : 滚动窗口大小，须满足 window > lag

        Returns
        -------
        pd.DataFrame : 值域 [-1, 1]

        Notes
        -----
        基于 pandas rolling.corr 实现全向量化计算，无显式循环。
        """
        if window <= lag:
            raise ValueError(
                f"window ({window}) 必须大于 lag ({lag})，"
                f"否则无法计算有效的自相关系数。"
            )
        return df.rolling(window=window, min_periods=window).corr(df.shift(lag))

    @staticmethod
    def Ts_Hurst(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        滚动 Hurst 指数（基于 R/S 分析的单尺度近似）。

        Hurst 指数 H 衡量时间序列的长程依赖性（分形维度）：
          - H ≈ 0.5 → 随机游走（布朗运动），无记忆性
          - H > 0.5 → 持续性趋势，过去涨则未来更可能继续涨
          - H < 0.5 → 反持续性（均值回归），过去涨则未来更可能跌

        R/S 分析（Rescaled Range Analysis）：
            1. 计算窗口均值 μ
            2. 累积偏差序列 Y(t) = Σ(X(i) - μ)
            3. 极差 R = max(Y) - min(Y)
            4. 标准差 S = std(X)
            5. H = log(R/S) / log(n/2)（单尺度近似，n = 窗口长度）

        Parameters
        ----------
        df     : 输入矩阵 (T × N)，通常为对数收益率序列（使用前请取 log return）
        window : 滚动窗口大小（建议 ≥ 20，越大越稳定）

        Returns
        -------
        pd.DataFrame : 值域理论为 (0, 1)，实际约在 [0.3, 0.8]

        Notes
        -----
        使用 rolling.apply(raw=True) 实现，`raw=True` 传递 numpy 数组
        而非 Series 对象，避免了 Python 对象开销（约快 10 倍）。
        """
        def _hurst_scalar(x: np.ndarray) -> float:
            """单窗口 R/S Hurst 估计。"""
            n = len(x)
            if n < 8:
                return np.nan
            # 过滤 NaN
            x = x[~np.isnan(x)]
            if len(x) < 8:
                return np.nan
            mean_x   = x.mean()
            deviation = np.cumsum(x - mean_x)
            R = deviation.max() - deviation.min()    # 极差
            S = x.std(ddof=1)                        # 标准差
            if S < 1e-10:
                return np.nan
            return np.log(R / S) / np.log(len(x) / 2)

        return df.rolling(
            window=window, min_periods=max(8, window // 2)
        ).apply(_hurst_scalar, raw=True)

    # ==================================================================
    # 量价因子 (Price-Volume)
    # ==================================================================

    @staticmethod
    def VWAP(
        close: pd.DataFrame,
        volume: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """
        成交量加权平均价（VWAP，Volume Weighted Average Price）。

        在滚动窗口内，以成交量为权重计算加权平均价格：
            VWAP = Σ(Pᵢ × Vᵢ) / ΣVᵢ，对 i ∈ [t-window+1, t]

        VWAP 常被视为短期均衡价格基准，高于 VWAP 时为卖压强信号，
        低于 VWAP 时为买盘强信号。

        Parameters
        ----------
        close  : 收盘价矩阵 (T × N)
        volume : 成交量矩阵 (T × N)，单位任意
        window : 滚动窗口大小

        Returns
        -------
        pd.DataFrame : VWAP 矩阵 (T × N)，量纲与 close 相同
        """
        pv = close * volume
        vwap = (
            pv.rolling(window=window, min_periods=window).sum()
            / volume.rolling(window=window, min_periods=window).sum()
        )
        return vwap

    @staticmethod
    def VWAP_Bias(
        close: pd.DataFrame,
        volume: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """
        VWAP 乖离率（VWAP Bias）。

        衡量当前收盘价相对滚动 VWAP 的百分比偏离程度：
            VWAP_Bias = Close / VWAP(window) - 1

        经济含义：
          - > 0：价格高于 VWAP，多头占优（或相对超买）
          - < 0：价格低于 VWAP，空头占优（或相对低估）
          - 均值回归信号：极端负乖离（大幅低于 VWAP）往往预示短期反弹
          - 趋势跟踪信号：持续正乖离的股票动量较强

        与 PVDeviation 的区别：
          - VWAP_Bias：百分比偏离（原始幅度，保留量纲）
          - PVDeviation：除以价格标准差标准化（无量纲）

        Parameters
        ----------
        close  : 收盘价矩阵 (T × N)
        volume : 成交量矩阵 (T × N)，单位任意
        window : 计算 VWAP 的滚动窗口大小

        Returns
        -------
        pd.DataFrame : VWAP 乖离率 (T × N)，值域无界，典型范围 [-0.1, 0.1]
                       VWAP 为 0 或 NaN 时对应位置返回 NaN
        """
        vwap = AlphaOps.VWAP(close, volume, window)
        vwap[vwap.abs() < 1e-10] = np.nan   # 避免除以零
        return close / vwap - 1.0

    @staticmethod
    def PVDeviation(
        close: pd.DataFrame,
        volume: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """
        量价背离指标（Price-Volume Deviation）。

        衡量当前收盘价与 VWAP 的标准化偏差：
            PVDev = (Close - VWAP) / RollingStd(Close, window)

        经济含义：
          - PVDev > 0：价格高于近期量加权均价（相对超买）
          - PVDev < 0：价格低于近期量加权均价（相对超卖）
          - 结合成交量分析：量缩价升（PVDev > 0 + 缩量）为强势信号

        Parameters
        ----------
        close  : 收盘价矩阵 (T × N)
        volume : 成交量矩阵 (T × N)
        window : 计算 VWAP 和标准差的滚动窗口

        Returns
        -------
        pd.DataFrame : 标准化量价偏差，值域无界（典型范围 [-3, 3]）
        """
        vwap    = AlphaOps.VWAP(close, volume, window)
        std_    = close.rolling(window=window, min_periods=window).std()
        std_[std_ < 1e-10] = np.nan
        return (close - vwap) / std_

    @staticmethod
    def Amihud(
        close: pd.DataFrame,
        volume: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """
        Amihud 非流动性因子（Amihud Illiquidity Ratio）。

        衡量单位成交量引起的价格冲击，是流动性的反向指标：
            Illiq = mean(|retₜ| / volumeₜ) for t in [t-window+1, t]

        其中 ret = close/close.shift(1) - 1（日收益率绝对值）。

        经济含义：
          - Illiq 越大 → 流动性越差，小额交易引起大幅价格波动
          - 常作为流动性溢价因子：非流动性高的股票要求更高预期收益
          - 微盘股 Illiq 通常远高于大盘蓝筹股

        Parameters
        ----------
        close  : 收盘价矩阵 (T × N)
        volume : 成交量矩阵 (T × N)，建议单位为"手（100股）"或"元"
        window : 滚动均值窗口

        Returns
        -------
        pd.DataFrame : Amihud 比率 (T × N)，值 ≥ 0

        Notes
        -----
        建议在使用前取对数变换 np.log(1 + Amihud) 以消除极端值。
        成交量为 0 的行对应位置返回 NaN。
        """
        ret     = close.pct_change()             # 日收益率
        safe_vol = volume.replace(0, np.nan)     # 避免除以零
        illiq   = (ret.abs() / safe_vol).rolling(
            window=window, min_periods=window
        ).mean()
        return illiq

    # ==================================================================
    # 动量因子 (Momentum)
    # ==================================================================

    @staticmethod
    def RiskAdjMomentum(
        close: pd.DataFrame,
        window: int,
        vol_window: int,
    ) -> pd.DataFrame:
        """
        风险调整动量因子（Risk-Adjusted Momentum）。

        经典动量因子（累计收益）除以同期波动率，衡量单位风险下的动量强度：
            RiskAdjMom = PctChange(window) / RollingStd(ret, vol_window)

        相比原始动量，风险调整动量：
          - 剔除了高波动率驱动的虚假动量信号
          - 对低波动率趋势股给予更高评分
          - 与 Sharpe 比率的横截面比较含义类似

        Parameters
        ----------
        close      : 收盘价矩阵 (T × N)
        window     : 动量计算周期（累计收益的回看窗口，天数）
        vol_window : 波动率计算窗口（天数），通常 vol_window < window

        Returns
        -------
        pd.DataFrame : 风险调整动量值，值域无界（典型范围 [-5, 5]）
        """
        ret      = close.pct_change()
        cum_ret  = close.pct_change(periods=window)                   # 累计收益
        roll_vol = ret.rolling(window=vol_window, min_periods=vol_window).std()
        roll_vol[roll_vol < 1e-10] = np.nan
        return cum_ret / roll_vol

    @staticmethod
    def PricePathQuality(
        close: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """
        路径质量因子（Price Path Quality）。

        衡量价格在 window 天内趋势的"单调性"与"线性程度"：
            PathQuality = |Spearman(t, price)| × Pearson(t, price)²

        - |Spearman(t, price)|：衡量价格时序的单调性（单调上涨/下跌越强越接近 1）
        - Pearson(t, price)²  ：衡量价格与时间的线性拟合优度（R²）
        - 两者乘积：同时要求趋势单调且线性（锯齿形趋势得分低）

        数值范围 [0, 1]：
          - 接近 1 → 价格在窗口内近似线性单调趋势（最优趋势路径）
          - 接近 0 → 震荡或无明显趋势

        Parameters
        ----------
        close  : 收盘价矩阵 (T × N)
        window : 评估窗口大小（建议 10~60 天）

        Returns
        -------
        pd.DataFrame : 路径质量分数 (T × N)，值域 [0, 1]

        Notes
        -----
        使用 rolling.apply(raw=True) + scipy.stats，
        raw=True 传递 numpy 数组，性能优于 raw=False 约 10 倍。
        """
        def _path_quality(x: np.ndarray) -> float:
            """计算单个窗口的路径质量分数。"""
            n = len(x)
            valid_mask = ~np.isnan(x)
            x_valid = x[valid_mask]
            if len(x_valid) < 4:
                return np.nan
            t = np.arange(len(x_valid), dtype=np.float64)
            # Spearman 单调性
            rho_result = spearmanr(t, x_valid)
            rho = rho_result.statistic if hasattr(rho_result, 'statistic') else rho_result[0]
            if np.isnan(rho):
                return np.nan
            # Pearson 线性 R²
            r_result = pearsonr(t, x_valid)
            r = r_result.statistic if hasattr(r_result, 'statistic') else r_result[0]
            if np.isnan(r):
                return np.nan
            return float(abs(rho) * r ** 2)

        return close.rolling(
            window=window, min_periods=max(4, window // 2)
        ).apply(_path_quality, raw=True)

    @staticmethod
    def RangeBreakout(
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """
        区间震荡突破因子（Range Breakout Position）。

        计算当前收盘价在过去 window 天高低区间内的相对位置（通道位置）：
            RangeBreakout = (Close - RollingMin(Low, window))
                           / (RollingMax(High, window) - RollingMin(Low, window))

        经济含义：
          - = 1.0 → 当前价格处于 window 日最高点（强势突破）
          - = 0.0 → 当前价格处于 window 日最低点（弱势跌破）
          - = 0.5 → 处于区间中位
          - 常用于趋势跟踪策略：值高 → 看多，值低 → 看空

        Parameters
        ----------
        close  : 收盘价矩阵 (T × N)
        high   : 最高价矩阵 (T × N)
        low    : 最低价矩阵 (T × N)
        window : 区间计算回看窗口（天数）

        Returns
        -------
        pd.DataFrame : 区间位置 (T × N)，值域 [0, 1]，
                       区间宽度接近 0 时返回 NaN
        """
        rol_max = high.rolling(window=window, min_periods=window).max()
        rol_min = low.rolling(window=window, min_periods=window).min()
        range_  = rol_max - rol_min
        range_[range_ < 1e-10] = np.nan
        return (close - rol_min) / range_

    # ==================================================================
    # 技术指标 (Technical Indicators)
    # ==================================================================

    @staticmethod
    def RSI(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        相对强弱指标（RSI，Relative Strength Index）。

        使用 Wilder 平滑法（等价于 EWM，span = 2×window - 1）计算 RSI：
            1. 计算日收益差分：delta = close.diff()
            2. 分离上涨（gain）与下跌（loss）
            3. Wilder 平滑：AvgGain = EWM(gain, span=2w-1)
                           AvgLoss = EWM(loss, span=2w-1)
            4. RS = AvgGain / AvgLoss
            5. RSI = 100 - 100 / (1 + RS)

        经济含义：
          - RSI > 70 → 超买区间，注意回调风险
          - RSI < 30 → 超卖区间，注意反弹机会
          - 与标准 SMA 版 RSI 不同，Wilder EWM 更平滑，避免突变

        Parameters
        ----------
        close  : 收盘价矩阵 (T × N)
        window : RSI 计算周期，标准值为 14 天

        Returns
        -------
        pd.DataFrame : RSI 值 (T × N)，值域 [0, 100]

        Notes
        -----
        全向量化实现（基于 pandas ewm），无显式股票循环。
        span = 2×window - 1 是 Wilder 平滑与 EWM 的等价变换关系。
        """
        delta  = close.diff()
        gain   = delta.clip(lower=0.0)          # 上涨部分，跌则为 0
        loss   = (-delta).clip(lower=0.0)       # 下跌绝对值，涨则为 0
        span   = 2 * window - 1                 # Wilder 平滑等价 EWM span
        avg_gain = gain.ewm(span=span, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(span=span, min_periods=window, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        rsi      = 100.0 - 100.0 / (1.0 + rs)
        return rsi

    @staticmethod
    def KDJ(
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        n: int = 9,
        m1: int = 3,
        m2: int = 3,
    ) -> pd.DataFrame:
        """
        KDJ 随机指标——返回 K 值（最常用于因子构建）。

        计算步骤：
            1. RSV（原始随机值）：
               RSV = (Close - Lowest_Low(n)) / (Highest_High(n) - Lowest_Low(n)) × 100
            2. K 值（K线）：K = EWM(RSV, span=2×m1-1)
               （等价于 Wilder 平滑：K = 2/3×K(prev) + 1/3×RSV，当 m1=3 时）
            3. D 值（信号线）：D = EWM(K, span=2×m2-1)  [未返回]
            4. J 值（背离）  ：J = 3×K - 2×D            [未返回]

        经济含义（K 值）：
          - K > 80 → 超买区间，趋势可能减弱
          - K < 20 → 超卖区间，存在反弹机会
          - K 线上穿 D 线 → 金叉，买入信号
          - K 线下穿 D 线 → 死叉，卖出信号

        Parameters
        ----------
        close : 收盘价矩阵 (T × N)
        high  : 最高价矩阵 (T × N)
        low   : 最低价矩阵 (T × N)
        n     : RSV 计算的回看天数，标准值 9
        m1    : K 值的 EWM 周期，标准值 3
        m2    : D 值的 EWM 周期（内部计算用），标准值 3

        Returns
        -------
        pd.DataFrame : K 值 (T × N)，值域 [0, 100]

        Notes
        -----
        全向量化实现，基于 pandas rolling + ewm，无显式股票循环。
        若需 D/J 值，可在外部调用：
            D = KDJ_K.ewm(span=2*m2-1, adjust=False).mean()
            J = 3 * KDJ_K - 2 * D
        """
        # 1. 计算 n 日最高价/最低价
        highest_high = high.rolling(window=n, min_periods=n).max()
        lowest_low   = low.rolling(window=n, min_periods=n).min()

        # 2. RSV：当前收盘在 n 日高低区间内的相对位置 × 100
        range_ = highest_high - lowest_low
        range_[range_ < 1e-10] = np.nan
        rsv = (close - lowest_low) / range_ * 100.0

        # 3. K 值：对 RSV 做 Wilder EWM 平滑（span = 2×m1 - 1）
        k_value = rsv.ewm(span=2 * m1 - 1, min_periods=n, adjust=False).mean()
        return k_value

    @staticmethod
    def MACD(
        close: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """
        MACD 柱状图因子（Moving Average Convergence/Divergence Histogram）。

        计算步骤：
            1. EMA_fast  = EMA(close, fast)     [快线，如 EMA(12)]
            2. EMA_slow  = EMA(close, slow)     [慢线，如 EMA(26)]
            3. MACD 线   = EMA_fast - EMA_slow  [差离值]
            4. Signal 线 = EMA(MACD 线, signal) [信号线]
            5. MACD 柱   = MACD 线 - Signal 线  [返回值，即 Histogram]

        经济含义（柱状图）：
          - 柱 > 0 → 快线在慢线上方，多头占优
          - 柱 < 0 → 快线在慢线下方，空头占优
          - 柱由负转正（零轴上穿）→ 金叉信号
          - 柱由正转负（零轴下穿）→ 死叉信号
          - 柱绝对值收缩 → 趋势减弱

        Parameters
        ----------
        close  : 收盘价矩阵 (T × N)
        fast   : 快速 EMA 周期，标准值 12
        slow   : 慢速 EMA 周期，标准值 26
        signal : 信号线 EMA 周期，标准值 9

        Returns
        -------
        pd.DataFrame : MACD 柱状图（Histogram） (T × N)，值域无界
                       正值表示上涨动能强，负值表示下跌动能强

        Notes
        -----
        全向量化实现（三次 ewm），无显式股票循环。
        若需 MACD 线或 Signal 线，可分步计算：
            macd_line   = close.ewm(span=fast).mean() - close.ewm(span=slow).mean()
            signal_line = macd_line.ewm(span=signal).mean()
        """
        ema_fast   = close.ewm(span=fast,   min_periods=fast,   adjust=False).mean()
        ema_slow   = close.ewm(span=slow,   min_periods=slow,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
        histogram  = macd_line - signal_line
        return histogram

    # ==================================================================
    # 缩量稳价因子 (VolSpike-PriceStable)
    # ==================================================================
    #
    # 核心逻辑：找出成交量（换手率）突然放大，但价格波动却异常收缩的股票。
    # 此类信号常见于主力资金在建仓阶段刻意压制价格波动的行为特征：
    #   - 大量成交发生，但买卖双方力量均衡，价格纹丝不动
    #   - 暗示水下筹码交换（洗盘/建仓），属于潜伏信号
    #
    # 三个子因子：
    #   VolSpike        — 换手率近期均值 / 历史均值（量能放大倍数）
    #   PriceVarShrink  — 价格方差近期 / 历史（波动收缩程度）
    #   PriceMeanShrink — 收盘价均值近期 / 历史（价格趋势下移程度）
    #
    # 复合因子：
    #   VolSpikeStablePrice — 同时满足量放大、价格方差收缩、均价下移的综合信号
    # ==================================================================

    @staticmethod
    def VolSpike(
        turnover: pd.DataFrame,
        n_short: int = 3,
        n_long: int = 10,
        n_multiple: float = 2.0,
    ) -> pd.DataFrame:
        """
        成交量（换手率）放大因子（Volume Spike Ratio）。

        衡量当前短期换手率相对历史基准的放大程度：
            VolSpike = M1 / M2

        其中：
            M1 = mean(turnover, n_short)              — 最近 n_short 天均换手率
            M2 = mean(turnover[t-n_short-n_long : t-n_short], n_long)
                                                       — 前 n_long 天均换手率（基准期）

        判定条件（量显著放大）：
            VolSpike > n_multiple（即 M1 > M2 × n_multiple）

        经济含义：
          - VolSpike >> 1 → 近期成交量相比历史大幅放大，市场活跃度激增
          - 结合价格波动收缩（PriceVarShrink）可识别"量增价稳"的建仓行为
          - 单独使用时也可作为市场情绪/活跃度的代理指标

        Parameters
        ----------
        turnover   : 换手率矩阵 (T × N)，值为每日换手率（如 0.02 表示 2%）
                     也可替换为成交量、成交额等量能代理指标
        n_short    : 短期观察窗口（天数），观察当前异动，默认 3 天
        n_long     : 长期基准窗口（天数），作为背景对比，默认 10 天
        n_multiple : 判定"显著放大"的倍数阈值（即 vol_mult 参数），默认 2.0
                     VolSpike > n_multiple 时视为量能显著放大

        Returns
        -------
        pd.DataFrame : 量能放大比率 (T × N)，值 ≥ 0
                       值域无界（典型范围 [0, 5]），前 n_short+n_long-1 行为 NaN
                       M2 ≤ 0 或为 NaN 时对应位置返回 NaN

        Notes
        -----
        基准期（M2）取 t 日往前推 n_short 天之外的 n_long 天，
        确保短期窗口（M1）与基准期（M2）完全不重叠，避免自相关。
        """
        # M1：最近 n_short 天均换手率
        m1 = turnover.rolling(window=n_short, min_periods=n_short).mean()

        # M2：基准期（向前跳过 n_short 天，再取 n_long 天）
        # 实现：先对换手率做 n_short 天延迟，再取 n_long 天均值
        m2 = turnover.shift(n_short).rolling(window=n_long, min_periods=n_long).mean()

        # 避免除以零或基准为零
        m2_safe = m2.copy()
        m2_safe[m2_safe.abs() < 1e-10] = np.nan

        return m1 / m2_safe

    @staticmethod
    def PriceVarShrink(
        close: pd.DataFrame,
        open_: pd.DataFrame,
        n_short: int = 3,
        n_long: int = 10,
        n_multiple: float = 0.5,
    ) -> pd.DataFrame:
        """
        价格方差收缩因子（Price Variance Shrink Ratio）。

        衡量近期价格波动相对历史基准的收缩程度：
            PriceVarShrink = V1 / V2

        其中，方差计算将收盘价与开盘价合并后联合计算（增大样本量，更稳健）：
            combined_short = [close_{t-n+1..t}, open_{t-n+1..t}]  合并为 2×n_short 个点
            combined_long  = [close_{t-n_s-n_l+1..t-n_s}, open_{...}]  2×n_long 个点
            V1 = var(combined_short)
            V2 = var(combined_long)

        判定条件（波动显著收缩）：
            PriceVarShrink < n_multiple（即 V1 < V2 × n_multiple）

        经济含义：
          - PriceVarShrink << 1 → 近期价格波动大幅收窄（无论开盘/收盘均如此）
          - 结合量能放大（VolSpike）构成"量大价稳"信号
          - 价格方差收缩越小，说明买卖双方在博弈中越趋于均衡（筹码交换）

        Parameters
        ----------
        close      : 收盘价矩阵 (T × N)
        open_      : 开盘价矩阵 (T × N)（注意：Python 保留字，参数名用 open_）
        n_short    : 短期方差计算窗口（天数），默认 3 天
        n_long     : 长期方差基准窗口（天数），默认 10 天
        n_multiple : 判定"显著收缩"的倍数阈值（即 pricevar_shrink 参数），默认 0.5
                     PriceVarShrink < n_multiple 时视为价格波动显著收缩

        Returns
        -------
        pd.DataFrame : 价格方差收缩比率 (T × N)，值 ≥ 0
                       值越小表示近期波动相对历史越收缩
                       V2 ≤ 0 或为 NaN 时对应位置返回 NaN

        Notes
        -----
        合并开盘价与收盘价后一起计算方差，等效于同时关注日内振幅与日间波动，
        使短窗口（如 3 天只有 6 个点）的估计更稳健。
        基准期同样跳过 n_short 天，与 VolSpike 逻辑保持一致，确保窗口不重叠。
        """
        def _rolling_var_combined(
            price_a: pd.DataFrame,
            price_b: pd.DataFrame,
            window: int,
            shift: int = 0,
        ) -> pd.DataFrame:
            """将两个价格序列合并后计算滚动方差（逐列 apply）。"""
            if shift > 0:
                price_a = price_a.shift(shift)
                price_b = price_b.shift(shift)

            def _col_var(col_a: pd.Series, col_b: pd.Series) -> pd.Series:
                """对单列合并序列求滚动方差。"""
                def _var_window(idx: int) -> float:
                    if idx < window - 1:
                        return np.nan
                    a_vals = col_a.iloc[idx - window + 1 : idx + 1].values
                    b_vals = col_b.iloc[idx - window + 1 : idx + 1].values
                    combined = np.concatenate([a_vals, b_vals])
                    valid = combined[~np.isnan(combined)]
                    if len(valid) < 3:
                        return np.nan
                    return float(np.var(valid, ddof=1))

                return pd.Series(
                    [_var_window(i) for i in range(len(col_a))],
                    index=col_a.index,
                )

            result = {}
            for col in price_a.columns:
                result[col] = _col_var(price_a[col], price_b[col])
            return pd.DataFrame(result, index=price_a.index)[price_a.columns]

        # V1：最近 n_short 天，合并 close + open 的方差
        v1 = _rolling_var_combined(close, open_, window=n_short, shift=0)

        # V2：基准期，向前跳过 n_short 天后取 n_long 天，合并 close + open 的方差
        v2 = _rolling_var_combined(close, open_, window=n_long, shift=n_short)

        # 避免除以零
        v2_safe = v2.copy()
        v2_safe[v2_safe.abs() < 1e-10] = np.nan

        return v1 / v2_safe

    @staticmethod
    def PriceMeanShrink(
        close: pd.DataFrame,
        n_short: int = 3,
        n_long: int = 10,
        n_multiple: float = 0.98,
    ) -> pd.DataFrame:
        """
        收盘价均值下移因子（Price Mean Shrink Ratio）。

        衡量近期收盘价均值相对历史基准的相对水平：
            PriceMeanShrink = P1 / P2

        其中：
            P1 = mean(close, n_short)              — 最近 n_short 天均价
            P2 = mean(close[t-n_short-n_long : t-n_short], n_long)
                                                    — 前 n_long 天均价（基准期）

        判定条件（价格均值相对下移）：
            PriceMeanShrink < n_multiple（即 P1 < P2 × n_multiple）

        经济含义：
          - PriceMeanShrink < 1 → 近期均价低于历史基准，股价处于下沉趋势
          - 与量能放大结合时，表现为"量增价跌"或"量增价横"中的价格滑落
          - 主力在打压吸筹时常见此形态：增量换手但均价小幅走低
          - 值域通常在 [0.9, 1.1] 附近；<1 为价格下移，>1 为价格上移

        Parameters
        ----------
        close      : 收盘价矩阵 (T × N)
        n_short    : 短期均值计算窗口（天数），默认 3 天
        n_long     : 长期基准窗口（天数），默认 10 天
        n_multiple : 判定"价格均值下移"的倍数阈值（即 price_shrink 参数），默认 0.98
                     PriceMeanShrink < n_multiple 时视为均价显著下移

        Returns
        -------
        pd.DataFrame : 价格均值比率 (T × N)，值 > 0
                       < 1 表示均价下移，> 1 表示均价上移，= 1 表示持平
                       P2 ≤ 0 或为 NaN 时对应位置返回 NaN

        Notes
        -----
        基准期同样跳过 n_short 天，与 VolSpike / PriceVarShrink 逻辑一致，
        三个子因子对应同一组时间窗口，可直接复合叠加。
        """
        # P1：最近 n_short 天均价
        p1 = close.rolling(window=n_short, min_periods=n_short).mean()

        # P2：基准期（向前跳过 n_short 天，再取 n_long 天均值）
        p2 = close.shift(n_short).rolling(window=n_long, min_periods=n_long).mean()

        # 避免除以零（价格序列通常 > 0，但防御性处理）
        p2_safe = p2.copy()
        p2_safe[p2_safe.abs() < 1e-10] = np.nan

        return p1 / p2_safe

    @staticmethod
    def VolSpikeStablePrice(
        turnover: pd.DataFrame,
        close: pd.DataFrame,
        open_: pd.DataFrame,
        n_short: int = 3,
        n_long: int = 10,
        n_multiple: float = 2.0,
        vol_mult: float = 2.0,
        pricevar_shrink: float = 0.5,
        price_shrink: float = 0.98,
    ) -> pd.DataFrame:
        """
        缩量稳价复合因子（Volume Spike + Stable Price Composite）。

        同时捕捉"换手率显著放大"、"价格方差显著收缩"、"均价小幅下移"三重信号，
        三者同时满足时得分更高，用于识别主力建仓/筹码交换行为。

        子因子计算：
            S1 = VolSpike(turnover, n_short, n_long)        — 量能放大比（M1/M2）
            S2 = PriceVarShrink(close, open_, n_short, n_long) — 价格方差收缩比（V1/V2）
            S3 = PriceMeanShrink(close, n_short, n_long)    — 均价下移比（P1/P2）

        各子因子信号方向：
            S1 越大（量越放大）→ 信号越强，乘以正权重
            S2 越小（波动越收缩）→ 信号越强，取倒数或负号
            S3 越小（价格越下移）→ 信号越强，取倒数或负号

        复合打分（三重归一化后加权）：
            综合信号 = w1 × ZScore(S1) + w2 × ZScore(-S2) + w3 × ZScore(-S3)

        其中各自的截止阈值（vol_mult / pricevar_shrink / price_shrink）决定
        各子因子是否真正"显著"——得分由原始比率直接计算，阈值参数供外部分析使用。

        Parameters
        ----------
        turnover        : 换手率矩阵 (T × N)
        close           : 收盘价矩阵 (T × N)
        open_           : 开盘价矩阵 (T × N)
        n_short         : 短期观察窗口（天数），默认 3 天
        n_long          : 长期基准窗口（天数），默认 10 天
        n_multiple      : 三个子因子共用的基础倍数参数，传入各子因子的 n_multiple
                          （当三个阈值需统一调整时使用；独立调整请分别设置下方参数）
        vol_mult        : 量能放大的判定阈值（VolSpike > vol_mult 为显著），默认 2.0
                          覆盖 n_multiple 对 VolSpike 子因子的影响
        pricevar_shrink : 价格方差收缩的判定阈值（PriceVarShrink < pricevar_shrink 为显著），
                          默认 0.5；覆盖 n_multiple 对 PriceVarShrink 子因子的影响
        price_shrink    : 均价下移的判定阈值（PriceMeanShrink < price_shrink 为显著），
                          默认 0.98；覆盖 n_multiple 对 PriceMeanShrink 子因子的影响

        Returns
        -------
        pd.DataFrame : 复合因子得分 (T × N)，值域无界（截面 ZScore 标准化后的线性组合）
                       正值表示"量增价稳"信号强度越高
                       前 n_short+n_long-1 行为 NaN

        Notes
        -----
        1. 三个子因子先做截面 ZScore 标准化，再等权相加，确保量纲统一。
        2. S2（价格方差比）取负号后 ZScore，使"波动越收缩 → 得分越高"。
        3. S3（均价比）取负号后 ZScore，使"均价越下移 → 得分越高"。
        4. 阈值参数（vol_mult / pricevar_shrink / price_shrink）不影响因子值的
           计算逻辑，仅作为策略层面的参考阈值，供外部条件筛选使用。
        5. 如需仅保留"三重信号同时满足"的股票，可在外部做如下掩码：
               mask = (S1 > vol_mult) & (S2 < pricevar_shrink) & (S3 < price_shrink)
               factor = composite.where(mask)
        """
        # ── 计算三个子因子 ──────────────────────────────────────────
        s1 = AlphaOps.VolSpike(
            turnover,
            n_short=n_short,
            n_long=n_long,
            n_multiple=vol_mult,
        )
        s2 = AlphaOps.PriceVarShrink(
            close,
            open_,
            n_short=n_short,
            n_long=n_long,
            n_multiple=pricevar_shrink,
        )
        s3 = AlphaOps.PriceMeanShrink(
            close,
            n_short=n_short,
            n_long=n_long,
            n_multiple=price_shrink,
        )

        # ── 截面 ZScore 标准化 + 方向对齐 ─────────────────────────
        # S1：量能越大 → 得分越高（正向）
        z1 = AlphaOps.ZScore(s1)

        # S2：波动越收缩（比率越小）→ 得分越高（取负后标准化）
        z2 = AlphaOps.ZScore(-s2)

        # S3：均价越下移（比率越小）→ 得分越高（取负后标准化）
        z3 = AlphaOps.ZScore(-s3)

        # ── 等权复合 ───────────────────────────────────────────────
        composite = (z1 + z2 + z3) / 3.0

        return composite
