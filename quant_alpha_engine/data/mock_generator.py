"""
MockDataGenerator
=================
生成仿真A股市场数据，用于 QuantAlpha_Engine 的测试与演示。

特性：
- 基于几何布朗运动(GBM)叠加行业共同因子，模拟真实相关结构
- 自动生成停牌/涨跌停状态矩阵
- 支持随机缺失期（新股上市延迟、退市等）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MockData:
    """存储所有模拟市场数据的容器。

    Attributes
    ----------
    close       : 收盘价矩阵 (T × N)，index=日期，columns=股票代码
    open        : 开盘价矩阵 (T × N)
    high        : 最高价矩阵 (T × N)
    low         : 最低价矩阵 (T × N)
    volume      : 成交量矩阵 (T × N)
    industry    : 行业映射 Series (index=股票代码，values=行业名称)
    is_suspended: 停牌状态矩阵 (T × N)，True 表示停牌
    is_limit    : 涨跌停状态矩阵 (T × N)，True 表示触发涨停或跌停
    """
    close:        pd.DataFrame
    open:         pd.DataFrame
    high:         pd.DataFrame
    low:          pd.DataFrame
    volume:       pd.DataFrame
    industry:     pd.Series
    is_suspended: pd.DataFrame
    is_limit:     pd.DataFrame


class MockDataGenerator:
    """工业级模拟数据生成器。

    Parameters
    ----------
    n_stocks     : 股票数量，默认 100
    n_days       : 交易日数量（约 2 年 = 504 天），默认 504
    n_industries : 行业数量，默认 10
    start_date   : 起始日期字符串，默认 '2022-01-01'
    seed         : 随机种子，None 则每次不同
    mu           : GBM 年化漂移率（基础），默认 0.08
    sigma        : GBM 年化波动率（基础），默认 0.30
    suspended_ratio : 历史停牌期概率，默认 0.04（约4%股票有停牌期）
    limit_pct    : 涨跌停触发阈值（绝对值），默认 0.099

    Examples
    --------
    >>> gen = MockDataGenerator(n_stocks=50, n_days=252, seed=42)
    >>> data = gen.generate()
    >>> data.close.shape
    (252, 50)
    """

    def __init__(
        self,
        n_stocks: int = 100,
        n_days: int = 504,
        n_industries: int = 10,
        start_date: str = "2022-01-01",
        seed: Optional[int] = 42,
        mu: float = 0.08,
        sigma: float = 0.30,
        suspended_ratio: float = 0.04,
        limit_pct: float = 0.099,
    ):
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.n_industries = n_industries
        self.start_date = start_date
        self.seed = seed
        self.mu = mu
        self.sigma = sigma
        self.suspended_ratio = suspended_ratio
        self.limit_pct = limit_pct

        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def generate(self) -> MockData:
        """生成完整的模拟市场数据集。

        Returns
        -------
        MockData
            包含 close/open/high/low/volume/industry/is_suspended/is_limit
        """
        dates = self._generate_trade_dates()
        stock_codes = [f"SH{60_0000 + i:04d}" for i in range(self.n_stocks)]
        industry_map = self._assign_industries(stock_codes)

        # ---- 生成收盘价（GBM + 行业共同因子）--------------------------
        close_arr = self._simulate_close(dates, industry_map)

        # ---- 日收益率 -> 推导其他价格字段 --------------------------------
        ret_arr = np.diff(np.log(close_arr), axis=0, prepend=np.log(close_arr[:1]))

        open_arr  = self._derive_open(close_arr, ret_arr)
        high_arr  = self._derive_high(open_arr, close_arr)
        low_arr   = self._derive_low(open_arr, close_arr)
        volume_arr = self._derive_volume(close_arr, ret_arr)

        # ---- 状态矩阵 --------------------------------------------------
        is_suspended_arr = self._generate_suspended(len(dates))
        # 停牌期间价格设为 NaN
        close_arr[is_suspended_arr]  = np.nan
        open_arr[is_suspended_arr]   = np.nan
        high_arr[is_suspended_arr]   = np.nan
        low_arr[is_suspended_arr]    = np.nan
        volume_arr[is_suspended_arr] = np.nan

        is_limit_arr = self._generate_limit(close_arr)

        # ---- 封装为 DataFrame -----------------------------------------
        idx = pd.DatetimeIndex(dates, name="date")
        col = pd.Index(stock_codes, name="stock")

        def _df(arr):
            return pd.DataFrame(arr, index=idx, columns=col)

        return MockData(
            close        = _df(close_arr),
            open         = _df(open_arr),
            high         = _df(high_arr),
            low          = _df(low_arr),
            volume       = _df(volume_arr),
            industry     = industry_map,
            is_suspended = pd.DataFrame(is_suspended_arr, index=idx, columns=col),
            is_limit     = pd.DataFrame(is_limit_arr,     index=idx, columns=col),
        )

    # ------------------------------------------------------------------
    # 内部生成方法
    # ------------------------------------------------------------------

    def _generate_trade_dates(self) -> list:
        """生成交易日序列（简单跳过周末，不处理节假日）。"""
        all_days = pd.bdate_range(start=self.start_date, periods=self.n_days)
        return all_days.tolist()

    def _assign_industries(self, stock_codes: list) -> pd.Series:
        """将股票随机分配至行业，保证每个行业至少有 2 只股票。"""
        industry_names = [f"行业{chr(0x4E00 + i)}" for i in range(self.n_industries)]
        # 先每个行业分配 2 只，剩余随机分配
        base = list(np.repeat(industry_names, 2))
        extra = self._rng.choice(industry_names, size=self.n_stocks - len(base)).tolist()
        labels = (base + extra)[:self.n_stocks]
        self._rng.shuffle(labels)
        return pd.Series(labels, index=stock_codes, name="industry")

    def _simulate_close(
        self, dates: list, industry_map: pd.Series
    ) -> np.ndarray:
        """
        使用 GBM 叠加行业共同因子生成收盘价矩阵。

        模型：
            dS_i = mu_i * dt + sigma_i * dW_i
            dW_i = rho * dW_industry_k + sqrt(1-rho^2) * dW_idio_i
        其中 rho = 0.5，行业内个股具有正相关性。
        """
        T, N = self.n_days, self.n_stocks
        dt = 1 / 252
        rho = 0.5
        rho_sqrt = np.sqrt(1 - rho ** 2)

        # 每只股票随机化 mu 和 sigma，增加异质性
        mus   = self.mu   + self._rng.uniform(-0.04, 0.04, size=N)
        sigmas = self.sigma + self._rng.uniform(-0.10, 0.10, size=N)
        sigmas = np.clip(sigmas, 0.10, 0.60)

        # 行业共同因子噪声 (T × n_industries)
        industry_noise = self._rng.standard_normal((T, self.n_industries))

        # 个股特质噪声 (T × N)
        idio_noise = self._rng.standard_normal((T, N))

        # 行业归属索引
        industry_names  = industry_map.unique().tolist()
        industry_idx    = industry_map.map({k: v for v, k in enumerate(industry_names)}).values  # (N,)

        # 合成收益率
        log_ret = np.zeros((T, N))
        for i in range(N):
            k = industry_idx[i]
            composite_noise = rho * industry_noise[:, k] + rho_sqrt * idio_noise[:, i]
            log_ret[:, i] = (mus[i] - 0.5 * sigmas[i] ** 2) * dt + sigmas[i] * np.sqrt(dt) * composite_noise

        # 初始价格随机化（10~100 元）
        init_price = self._rng.uniform(10, 100, size=N)
        log_price = np.cumsum(log_ret, axis=0) + np.log(init_price)
        close_arr = np.exp(log_price)
        return close_arr.astype(np.float32)

    def _derive_open(self, close: np.ndarray, log_ret: np.ndarray) -> np.ndarray:
        """开盘价 ≈ 昨收 * (1 + 小随机跳空)"""
        noise = self._rng.uniform(-0.005, 0.005, size=close.shape)
        open_arr = np.roll(close, 1, axis=0) * (1 + noise)
        open_arr[0] = close[0] * (1 + self._rng.uniform(-0.01, 0.01, size=close.shape[1]))
        return open_arr.astype(np.float32)

    def _derive_high(self, open_arr: np.ndarray, close: np.ndarray) -> np.ndarray:
        """最高价 = max(open, close) * (1 + 上影线)"""
        upper_shadow = self._rng.uniform(0.001, 0.025, size=close.shape)
        high_arr = np.maximum(open_arr, close) * (1 + upper_shadow)
        return high_arr.astype(np.float32)

    def _derive_low(self, open_arr: np.ndarray, close: np.ndarray) -> np.ndarray:
        """最低价 = min(open, close) * (1 - 下影线)"""
        lower_shadow = self._rng.uniform(0.001, 0.025, size=close.shape)
        low_arr = np.minimum(open_arr, close) * (1 - lower_shadow)
        return low_arr.astype(np.float32)

    def _derive_volume(self, close: np.ndarray, log_ret: np.ndarray) -> np.ndarray:
        """
        成交量与绝对收益率正相关（量价关系），加入随机噪声。
        单位：手（100股）。
        """
        T, N = close.shape
        base_vol = self._rng.uniform(5_000, 50_000, size=N)  # 基础日均成交量
        abs_ret  = np.abs(log_ret)
        # 波动越大，成交量越大（弹性系数 ~2）
        vol_multiplier = 1 + 2 * (abs_ret / (abs_ret.mean(axis=0, keepdims=True) + 1e-8))
        noise = self._rng.lognormal(0, 0.3, size=(T, N))
        volume_arr = base_vol[np.newaxis, :] * vol_multiplier * noise
        return volume_arr.astype(np.float32)

    def _generate_suspended(self, n_days: int) -> np.ndarray:
        """
        生成停牌状态矩阵。
        约 suspended_ratio 的股票会有 1~3 段连续停牌期，每段 5~20 个交易日。
        """
        mask = np.zeros((n_days, self.n_stocks), dtype=bool)
        n_suspended_stocks = max(1, int(self.n_stocks * self.suspended_ratio))
        stock_indices = self._rng.choice(self.n_stocks, size=n_suspended_stocks, replace=False)

        for s in stock_indices:
            n_periods = self._rng.integers(1, 4)  # 1~3 段停牌
            for _ in range(n_periods):
                start = self._rng.integers(0, n_days - 20)
                length = self._rng.integers(5, 21)
                end = min(start + length, n_days)
                mask[start:end, s] = True
        return mask

    def _generate_limit(self, close: np.ndarray) -> np.ndarray:
        """
        根据日涨跌幅判断涨跌停。
        close 中 NaN（停牌）不触发涨跌停。
        """
        # 前收盘（T×N）
        prev_close = np.roll(close, 1, axis=0)
        prev_close[0] = close[0]

        with np.errstate(invalid='ignore', divide='ignore'):
            pct_chg = (close - prev_close) / prev_close

        is_limit = np.abs(pct_chg) >= self.limit_pct
        # 停牌（NaN）不算涨跌停
        is_limit = is_limit & ~np.isnan(close)
        return is_limit
