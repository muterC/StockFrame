"""
RepositoryDataLoader
====================
从 DataRepository 加载真实 A 股市场数据，输出与 MockData 完全兼容的
MarketData 容器，可直接替换 MockDataGenerator 用于 AlphaOps + VectorEngine。

数据流：
    DataRepository (CSV 文件)
        └─► RepositoryDataLoader.load()
                └─► MarketData（T×N 宽表矩阵，与 MockData 完全同构）

使用示例：
    from data_loader import RepositoryDataLoader
    from quant_alpha_engine.ops import AlphaOps as op
    from quant_alpha_engine.backtest import VectorEngine

    loader = RepositoryDataLoader("rawdata/repository")
    data = loader.load(
        symbols    = ['600519', '000001', '000002'],
        start_date = '2024-01-01',
        end_date   = '2026-02-22',
        adj_type   = 'hfq',        # 'hfq' | 'qfq' | 'raw'
    )

    # 与 MockData 完全相同的使用方式
    factor = op.Rank(op.Ts_Delta(data.close, 20))
    result = VectorEngine(
        factor       = factor,
        close        = data.close,
        is_suspended = data.is_suspended,
        is_limit     = data.is_limit,
    ).run()

复权说明：
    - 'hfq'：后复权（推荐）。price_hfq = price_raw × hfq_factor。
             因子间可跨期直接比较，hfq_factor 单调递增。
    - 'qfq'：前复权。price_qfq = price_raw × (hfq_factor / hfq_factor.iloc[-1])。
             从已存储的 hfq 动态推导，避免存储 qfq 快照过期的问题。
    - 'raw'：不复权，使用原始价格。
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==================== 数据容器（与 MockData 完全同构）====================

@dataclass
class MarketData:
    """
    真实市场数据容器，与 MockData 完全同构，可直接替换用于 AlphaOps + VectorEngine。

    所有矩阵维度均为 (T × N)：
        - index   = pd.DatetimeIndex，名称 "date"（naive datetime，无时区）
        - columns = pd.Index，名称 "stock"（股票代码字符串）

    Attributes
    ----------
    close        : 收盘价矩阵（已复权，取决于 adj_type）
    open         : 开盘价矩阵（已复权）
    high         : 最高价矩阵（已复权）
    low          : 最低价矩阵（已复权）
    volume       : 成交量矩阵（不参与复权）
    vwap         : 成交均价矩阵（已复权）；若原始数据无 vwap 则为 None
    hfq_factor   : 后复权因子矩阵（始终保留，供下游动态推导 qfq）
    industry     : 行业映射 Series（index=股票代码，values=行业名称）；
                   若仓库无行业数据则填 "Unknown"
    is_suspended : 停牌状态矩阵（bool，True=停牌）
    is_limit     : 涨跌停状态矩阵（bool，True=触发涨跌停）
    adj_type     : 本次加载使用的复权方式（'hfq' / 'qfq' / 'raw'）
    start_date   : 实际加载的起始日期
    end_date     : 实际加载的截止日期
    symbols      : 实际加载的股票代码列表（有效股票，已排除全空股）
    missing_symbols : 请求但未找到数据的股票代码列表
    """
    # 价格 / 量 矩阵
    close:        pd.DataFrame
    open:         pd.DataFrame
    high:         pd.DataFrame
    low:          pd.DataFrame
    volume:       pd.DataFrame

    # 复权因子（始终保留后复权，供下游动态推导）
    hfq_factor:   pd.DataFrame

    # 辅助矩阵
    is_suspended: pd.DataFrame
    is_limit:     pd.DataFrame

    # 行业/元信息
    industry:     pd.Series

    # 可选字段
    vwap:         Optional[pd.DataFrame] = None

    # 元信息
    adj_type:        str       = 'hfq'
    start_date:      str       = ''
    end_date:        str       = ''
    symbols:         List[str] = field(default_factory=list)
    missing_symbols: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        t, n = self.close.shape
        return (
            f"MarketData("
            f"T={t}, N={n}, "
            f"adj={self.adj_type}, "
            f"dates={self.start_date}~{self.end_date}, "
            f"missing={len(self.missing_symbols)})"
        )

    def summary(self) -> str:
        """打印数据集概要信息。"""
        t, n = self.close.shape
        susp_pct  = self.is_suspended.values.mean() * 100
        limit_pct = self.is_limit.values.mean() * 100
        nan_pct   = self.close.isnull().values.mean() * 100

        lines = [
            "=" * 55,
            "  MarketData 概要",
            "=" * 55,
            f"  交易日数 × 股票数 : {t} × {n}",
            f"  日期范围           : {self.close.index[0].date()} ~ {self.close.index[-1].date()}",
            f"  复权方式           : {self.adj_type}",
            f"  NaN 比例 (close)   : {nan_pct:.2f}%",
            f"  停牌天数比例       : {susp_pct:.2f}%",
            f"  涨跌停天数比例     : {limit_pct:.2f}%",
        ]
        if self.missing_symbols:
            lines.append(f"  缺失股票 ({len(self.missing_symbols)})  : {self.missing_symbols}")
        if self.industry is not None:
            n_ind = self.industry.nunique()
            lines.append(f"  行业数量           : {n_ind}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def print_summary(self):
        """打印数据集概要信息（同 print(data.summary())）。"""
        print(self.summary())


# ==================== 加载器主类 ====================

class RepositoryDataLoader:
    """
    从 DataRepository 加载真实 A 股数据，输出 MarketData 容器。

    Parameters
    ----------
    repo_dir : str | Path
        DataRepository 根目录（与 DataRepository(base_dir=...) 一致）。
    limit_up_pct : float
        涨跌停判断阈值（绝对值），默认 0.099（科创板/创业板可改为 0.199）。

    Notes
    -----
    - 未找到数据的股票记录在 MarketData.missing_symbols，不影响其他股票。
    - 停牌识别：当天 volume == 0 或 close 为 NaN，则视为停牌。
    - 涨跌停识别：当日涨跌幅 >= limit_up_pct（基于后复权收盘价计算）。
    """

    def __init__(
        self,
        repo_dir: Union[str, Path] = "rawdata/repository",
        limit_up_pct: float = 0.099,
    ):
        # 延迟导入，避免循环依赖
        from data_process.repository import DataRepository

        self.repo = DataRepository(base_dir=repo_dir)
        self.limit_up_pct = limit_up_pct
        self.logger = logging.getLogger('RepositoryDataLoader')

    # ==================== 公开接口 ====================

    def load(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adj_type: str = 'hfq',
        fill_suspended: bool = True,
        min_trading_days: int = 20,
    ) -> MarketData:
        """
        加载市场数据并输出 MarketData 容器。

        Parameters
        ----------
        symbols : list[str] | None
            股票代码列表。None 时自动读取仓库中所有有 OHLCV 数据的股票。
        start_date : str | None
            起始日期，格式 'YYYY-MM-DD'。None 时取仓库最早日期。
        end_date : str | None
            截止日期，格式 'YYYY-MM-DD'。None 时取今天。
        adj_type : str
            复权方式：
            - 'hfq'：后复权（推荐，price × hfq_factor）
            - 'qfq'：前复权（price × hfq_factor / hfq_factor.iloc[-1]）
            - 'raw'：不复权
        fill_suspended : bool
            停牌日价格填充策略。True 时用前值 ffill 填充（停牌期间
            保持价格不变），is_suspended 矩阵仍标记为 True。
            False 时停牌日保留 NaN。
        min_trading_days : int
            最少有效交易日数，低于此阈值的股票从结果中剔除（默认 20 天）。

        Returns
        -------
        MarketData
        """
        if adj_type not in ('hfq', 'qfq', 'raw'):
            raise ValueError(f"adj_type 必须为 'hfq'/'qfq'/'raw'，收到: {adj_type!r}")

        # 1. 确定股票列表
        if symbols is None:
            symbols = self.repo.list_symbols('ohlcv')
        if not symbols:
            raise RuntimeError("仓库中没有 OHLCV 数据，请先运行 data_download.ipynb")

        # 2. 确定日期范围
        end_date = end_date or pd.Timestamp.today().strftime('%Y-%m-%d')

        # 3. 逐股加载原始数据
        ohlcv_dict:  Dict[str, pd.DataFrame] = {}
        adj_dict:    Dict[str, pd.DataFrame] = {}
        missing: List[str] = []

        for sym in symbols:
            try:
                ohlcv = self.repo.load_ohlcv_data(sym, start_date, end_date)
                if ohlcv.empty:
                    missing.append(sym)
                    continue
                ohlcv_dict[sym] = ohlcv

                # 尝试加载复权因子（raw 模式也加载，用于 is_limit 计算）
                try:
                    adj = self.repo.load_adjustment_factors(sym)
                    # 裁剪到与 ohlcv 相同的日期范围（宽松匹配，用 reindex）
                    adj_dict[sym] = adj
                except RuntimeError:
                    # 无复权数据时：adj 列全填 1.0
                    adj_dict[sym] = pd.DataFrame(
                        {'hfq_factor': 1.0, 'qfq_factor': 1.0},
                        index=ohlcv.index
                    )
                    self.logger.warning("[%s] 无复权数据，使用 hfq_factor=1.0", sym)

            except RuntimeError:
                missing.append(sym)
                self.logger.warning("[%s] 无 OHLCV 数据，已跳过", sym)

        if not ohlcv_dict:
            raise RuntimeError(f"指定的股票均无数据: {symbols}")

        if missing:
            self.logger.warning("缺失数据股票 (%d): %s", len(missing), missing)

        # 4. 构建统一日期索引（所有股票的并集交易日）
        all_dates = self._build_date_index(ohlcv_dict, start_date, end_date)

        # 5. 组装宽表矩阵
        raw_matrices = self._assemble_matrices(ohlcv_dict, all_dates)

        # 6. 组装后复权因子矩阵（与价格矩阵对齐）
        hfq_matrix = self._assemble_hfq_matrix(adj_dict, all_dates, raw_matrices['close'])

        # 7. 应用复权
        adj_matrices = self._apply_adjustment(raw_matrices, hfq_matrix, adj_type)

        # 8. 剔除有效交易日不足的股票
        valid_days = (~adj_matrices['close'].isnull()).sum(axis=0)
        too_short = valid_days[valid_days < min_trading_days].index.tolist()
        if too_short:
            self.logger.warning(
                "以下股票有效交易日 < %d，已剔除: %s", min_trading_days, too_short
            )
            for col in too_short:
                for m in adj_matrices.values():
                    if col in m.columns:
                        m.drop(columns=[col], inplace=True)
                if col in hfq_matrix.columns:
                    hfq_matrix.drop(columns=[col], inplace=True)
            missing.extend(too_short)

        valid_symbols = list(adj_matrices['close'].columns)
        if not valid_symbols:
            raise RuntimeError("剔除无效股票后，没有剩余股票可加载")

        # 9. 生成停牌 / 涨跌停矩阵
        is_suspended = self._build_suspended_matrix(raw_matrices['close'], raw_matrices['volume'])
        is_limit     = self._build_limit_matrix(adj_matrices['close'])

        # 10. 对停牌日价格进行 ffill 填充（可选）
        if fill_suspended:
            for key in ('close', 'open', 'high', 'low'):
                adj_matrices[key] = adj_matrices[key].ffill()
            hfq_matrix = hfq_matrix.ffill()

        # 11. 加载行业数据
        industry = self._load_industry(valid_symbols)

        # 12. 组装 vwap（若 ohlcv 中有 vwap 列）
        vwap_matrix = self._assemble_optional_matrix(ohlcv_dict, 'vwap', all_dates, hfq_matrix, adj_type)

        # 13. 统一列名 / 索引 name
        date_idx = all_dates.rename("date")
        stock_idx = pd.Index(valid_symbols, name="stock")

        def _reindex(df: pd.DataFrame) -> pd.DataFrame:
            return df.reindex(index=date_idx, columns=stock_idx)

        actual_start = date_idx[0].strftime('%Y-%m-%d')
        actual_end   = date_idx[-1].strftime('%Y-%m-%d')

        self.logger.info(
            "加载完成：%d 只股票 × %d 个交易日（%s ~ %s），复权=%s",
            len(valid_symbols), len(date_idx), actual_start, actual_end, adj_type
        )

        return MarketData(
            close        = _reindex(adj_matrices['close']),
            open         = _reindex(adj_matrices['open']),
            high         = _reindex(adj_matrices['high']),
            low          = _reindex(adj_matrices['low']),
            volume       = _reindex(raw_matrices['volume']),   # 量不复权
            hfq_factor   = _reindex(hfq_matrix),
            is_suspended = _reindex(is_suspended).fillna(False).astype(bool),
            is_limit     = _reindex(is_limit).fillna(False).astype(bool),
            industry     = industry,
            vwap         = _reindex(vwap_matrix) if vwap_matrix is not None else None,
            adj_type     = adj_type,
            start_date   = actual_start,
            end_date     = actual_end,
            symbols      = valid_symbols,
            missing_symbols = missing,
        )

    def load_financial(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        加载财务指标数据（季报粒度）。

        Returns
        -------
        dict[str, pd.DataFrame] | None
            key = 列名（如 'roe', 'revenue' 等）
            value = (T × N) 宽表，index=报告期，columns=股票代码
            无数据时返回 None。
        """
        if symbols is None:
            symbols = self.repo.list_symbols('financial')
        if not symbols:
            return None

        fin_dict: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.repo.load_financial_data(sym, start_date, end_date)
                if not df.empty:
                    fin_dict[sym] = df
            except RuntimeError:
                pass

        if not fin_dict:
            return None

        # 宽化：每列一个财务指标
        all_dates = sorted(set().union(*[set(df.index) for df in fin_dict.values()]))
        all_cols  = sorted(set().union(*[set(df.columns) for df in fin_dict.values()]))
        date_idx  = pd.DatetimeIndex(all_dates, name="date")

        result = {}
        for col in all_cols:
            wide = pd.DataFrame(index=date_idx, columns=list(fin_dict.keys()), dtype=float)
            wide.index.name = "date"
            wide.columns.name = "stock"
            for sym, df in fin_dict.items():
                if col in df.columns:
                    sym_series = df[col].reindex(date_idx)
                    wide[sym] = sym_series
            result[col] = wide

        return result

    def load_market_value(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        加载市值数据。

        Returns
        -------
        dict[str, pd.DataFrame] | None
            key = 'total_market_cap' / 'circulating_market_cap'
            value = (T × N) 宽表，index=日期，columns=股票代码
        """
        if symbols is None:
            symbols = self.repo.list_symbols('market_value')
        if not symbols:
            return None

        mv_dict: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.repo.load_market_value_data(sym, start_date, end_date)
                if not df.empty:
                    mv_dict[sym] = df
            except RuntimeError:
                pass

        if not mv_dict:
            return None

        # 统一日期索引
        all_dates = sorted(set().union(*[set(df.index) for df in mv_dict.values()]))
        date_idx = pd.DatetimeIndex(all_dates, name="date")
        all_cols = sorted(set().union(*[set(df.columns) for df in mv_dict.values()]))

        result = {}
        for col in all_cols:
            wide = pd.DataFrame(index=date_idx, columns=list(mv_dict.keys()), dtype=float)
            wide.index.name = "date"
            wide.columns.name = "stock"
            for sym, df in mv_dict.items():
                if col in df.columns:
                    wide[sym] = df[col].reindex(date_idx)
            result[col] = wide

        return result

    # ==================== 内部辅助方法 ====================

    @staticmethod
    def _build_date_index(
        ohlcv_dict: Dict[str, pd.DataFrame],
        start_date: Optional[str],
        end_date: str,
    ) -> pd.DatetimeIndex:
        """
        取所有股票 OHLCV 日期的并集，裁剪至 [start_date, end_date]，
        确保为 naive datetime（无时区）并排序。
        """
        all_dates = set()
        for df in ohlcv_dict.values():
            idx = df.index
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            all_dates.update(idx.tolist())

        date_idx = pd.DatetimeIndex(sorted(all_dates))

        if start_date:
            date_idx = date_idx[date_idx >= start_date]
        date_idx = date_idx[date_idx <= end_date]

        return date_idx

    @staticmethod
    def _assemble_matrices(
        ohlcv_dict: Dict[str, pd.DataFrame],
        all_dates: pd.DatetimeIndex,
    ) -> Dict[str, pd.DataFrame]:
        """将各股 OHLCV 序列组装为 (T × N) 宽表。"""
        symbols = list(ohlcv_dict.keys())
        matrices = {col: pd.DataFrame(index=all_dates, columns=symbols, dtype=float)
                    for col in ('open', 'high', 'low', 'close', 'volume')}

        for sym, df in ohlcv_dict.items():
            # 统一去时区
            idx = df.index
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            df = df.copy()
            df.index = idx

            for col in ('open', 'high', 'low', 'close', 'volume'):
                if col in df.columns:
                    matrices[col][sym] = df[col].reindex(all_dates)

        for m in matrices.values():
            m.index.name = "date"
            m.columns.name = "stock"

        return matrices

    def _assemble_hfq_matrix(
        self,
        adj_dict: Dict[str, pd.DataFrame],
        all_dates: pd.DatetimeIndex,
        close_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        将各股 hfq_factor 序列组装为 (T × N) 宽表，并对齐至价格日期索引。

        对齐策略：
          - 复权因子日期以交易日历（OHLCV）为准
          - 先 reindex 到 all_dates，再 ffill（处理节假日等缺失日期）
          - 若某日期早于复权数据最早日期，bfill 填充首日因子值
        """
        symbols = list(adj_dict.keys())
        hfq_wide = pd.DataFrame(index=all_dates, columns=symbols, dtype=float)
        hfq_wide.index.name = "date"
        hfq_wide.columns.name = "stock"

        for sym, adj in adj_dict.items():
            idx = adj.index
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            adj = adj.copy()
            adj.index = idx

            if 'hfq_factor' in adj.columns:
                series = adj['hfq_factor'].reindex(all_dates).ffill().bfill()
            else:
                series = pd.Series(1.0, index=all_dates)
                self.logger.warning("[%s] 复权数据缺少 hfq_factor 列，填充 1.0", sym)

            hfq_wide[sym] = series

        # 兜底：仍为 NaN 的位置填 1.0（上市前 / 数据缺口）
        hfq_wide = hfq_wide.fillna(1.0)
        return hfq_wide

    @staticmethod
    def _apply_adjustment(
        raw_matrices: Dict[str, pd.DataFrame],
        hfq_matrix: pd.DataFrame,
        adj_type: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        对价格矩阵应用复权。量 (volume) 不参与复权。

        - hfq：price_hfq = price_raw × hfq_factor
        - qfq：price_qfq = price_raw × (hfq_factor / hfq_factor.iloc[-1])
               从 hfq 动态推导，避免存储 qfq 快照过期。
        - raw：不处理，原样返回
        """
        price_cols = ('open', 'high', 'low', 'close')
        result = {}

        if adj_type == 'raw':
            for col in price_cols:
                result[col] = raw_matrices[col].copy()
            return result

        if adj_type == 'hfq':
            factor = hfq_matrix
        else:  # qfq
            # 取各股最新一行的 hfq_factor 作为分母
            last_hfq = hfq_matrix.ffill().iloc[-1]   # Series (N,)
            # 广播：factor[t, i] = hfq_matrix[t, i] / last_hfq[i]
            factor = hfq_matrix.div(last_hfq, axis='columns')

        for col in price_cols:
            result[col] = raw_matrices[col] * factor

        return result

    @staticmethod
    def _build_suspended_matrix(
        close_raw: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        停牌识别规则（满足任一即为停牌）：
          1. close 为 NaN（数据缺失，通常表示停牌/未上市）
          2. volume == 0（有挂牌价格但无成交量，A 股典型停牌模式）

        注：is_limit（涨跌停）时仍有成交，不视为停牌。
        """
        no_close  = close_raw.isnull()
        no_volume = (volume == 0).fillna(False)
        return (no_close | no_volume).astype(bool)

    def _build_limit_matrix(self, close_adj: pd.DataFrame) -> pd.DataFrame:
        """
        涨跌停识别：当日涨跌幅绝对值 >= limit_up_pct（基于复权收盘价）。
        首日无前收，视为未触发。
        """
        prev_close = close_adj.shift(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pct_chg = (close_adj - prev_close) / prev_close.abs()

        is_limit = pct_chg.abs() >= self.limit_up_pct
        is_limit.iloc[0] = False          # 首行：无前收，不判断
        is_limit = is_limit & close_adj.notna()  # 停牌日不算涨跌停
        return is_limit.fillna(False).astype(bool)

    def _load_industry(self, symbols: List[str]) -> pd.Series:
        """
        加载行业分类。优先从 reference/industry_classification.csv 读取申万一级行业，
        若文件不存在则填充 'Unknown'。

        CSV 格式（import_bloomberg_data.ipynb 生成）：
            columns: symbol, name, sw1_name, sw2_name, sw3_name, sw3_code

        Returns
        -------
        pd.Series
            index = 股票代码，values = 申万一级行业名称，name = 'industry'
        """
        industry_file = self.repo.reference_dir / "industry_classification.csv"
        if industry_file.exists():
            try:
                # symbol 列是普通列（非 index），dtype=str 防止前导0被截断
                industry_df = pd.read_csv(industry_file, dtype={'symbol': str}, encoding='utf-8')

                # 优先用申万一级；兼容旧格式的其他列名
                col_candidates = ['sw1_name', 'industry', '行业', 'industry_name', 'sector']
                industry_col = next((c for c in col_candidates if c in industry_df.columns), None)

                if industry_col and 'symbol' in industry_df.columns:
                    # 建立 symbol → 行业 的映射字典
                    mapping = industry_df.set_index('symbol')[industry_col].to_dict()
                    result = pd.Series(
                        {sym: mapping.get(sym, 'Unknown') for sym in symbols},
                        name='industry'
                    )
                    result.index.name = 'stock'
                    return result
            except Exception as e:
                self.logger.warning("行业分类文件读取失败: %s，填充 Unknown", e)

        # 无行业数据：全填 Unknown
        result = pd.Series('Unknown', index=symbols, name='industry')
        result.index.name = 'stock'
        return result

    def _assemble_optional_matrix(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
        col_name: str,
        all_dates: pd.DatetimeIndex,
        hfq_matrix: pd.DataFrame,
        adj_type: str,
    ) -> Optional[pd.DataFrame]:
        """
        组装可选价格列（如 vwap），若所有股票均无此列则返回 None。
        """
        has_col = any(col_name in df.columns for df in ohlcv_dict.values())
        if not has_col:
            return None

        raw = {col_name: pd.DataFrame(index=all_dates,
                                       columns=list(ohlcv_dict.keys()), dtype=float)}
        raw[col_name].index.name = "date"
        raw[col_name].columns.name = "stock"

        for sym, df in ohlcv_dict.items():
            if col_name in df.columns:
                idx = df.index
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                raw[col_name][sym] = df[col_name].reindex(all_dates)

        adj = self._apply_adjustment(
            {col_name: raw[col_name], 'open': raw[col_name],
             'high': raw[col_name], 'low': raw[col_name]},
            hfq_matrix, adj_type
        )
        return adj[col_name]
