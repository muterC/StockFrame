"""
AlphaOps — QuantAlpha_Engine 算子库
=====================================
所有算子均支持 pandas.DataFrame 向量化运算。
约定：DataFrame 的 Index 为时间，Columns 为股票代码。

算子分类
--------
时序类 (Time-Series)
    Ts_Sum, Ts_Mean, Ts_Max, Ts_Min, Ts_Delta, Ts_Delay,
    Ts_Std, Ts_Rank, Ts_Corr

截面类 (Cross-Sectional)
    Rank, ZScore, Scale

特殊类 (Special)
    Decay_Linear, Neutralize

用法示例
--------
>>> from quant_alpha_engine.ops import AlphaOps as op
>>> factor = op.Rank(op.Ts_Delta(close, 20))
>>> factor2 = op.Neutralize(op.Rank(op.Ts_Corr(volume, close, 10)), industry)
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import pandas as pd

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
