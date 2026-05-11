"""AkShare provider."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import pandas as pd

from ..symbols import as_timestamp, normalize_stock_code, to_akshare_symbol
from .base import BaseProvider


class AkShareProvider(BaseProvider):
    name = "akshare"
    daily_adjustments = ("", "qfq", "hfq")
    daily_direct_fields = (
        "date",
        "stock_code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change",
        "turnover",
    )

    def fetch_daily(self, code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        return self.fetch_daily_adjusted(code, start_date, end_date, "")

    def fetch_daily_adjusted(self, code: str, start_date: pd.Timestamp, end_date: pd.Timestamp, adjustment: str = "") -> pd.DataFrame:
        import akshare as ak

        raw = ak.stock_zh_a_hist(
            symbol=to_akshare_symbol(code),
            period="daily",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            adjust=adjustment or "",
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        frame = raw.rename(
            columns={
                "日期": "date",
                "股票代码": "stock_code",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "换手率": "turnover",
            }
        )
        frame["stock_code"] = code
        return frame

    def fetch_minute(
        self,
        code: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "1m",
        adjust: str = "",
    ) -> pd.DataFrame:
        import akshare as ak

        period_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"}
        start = as_timestamp(start_date).strftime("%Y-%m-%d %H:%M:%S") if start_date is not None else "1979-09-01 09:30:00"
        end = as_timestamp(end_date).strftime("%Y-%m-%d %H:%M:%S") if end_date is not None else "2222-01-01 15:00:00"
        raw = ak.stock_zh_a_hist_min_em(
            symbol=to_akshare_symbol(code),
            start_date=start,
            end_date=end,
            period=period_map.get(period, period.replace("m", "")),
            adjust=adjust,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        frame = raw.rename(
            columns={
                "时间": "datetime",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "均价": "vwap",
            }
        )
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame["stock_code"] = code
        frame["source"] = self.name
        return frame.sort_values("datetime").reset_index(drop=True)

    def fetch_realtime(self, codes: Sequence[str]) -> pd.DataFrame:
        import akshare as ak

        raw = ak.stock_zh_a_spot_em()
        if raw is None or raw.empty:
            return pd.DataFrame()
        frame = raw.rename(
            columns={
                "代码": "stock_code",
                "名称": "name",
                "最新价": "price",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "最高": "high",
                "最低": "low",
                "今开": "open",
                "昨收": "pre_close",
                "量比": "volume_ratio",
                "换手率": "turnover",
                "市盈率-动态": "pe_ttm",
                "市净率": "pb",
                "总市值": "total_market_cap",
                "流通市值": "float_market_cap",
            }
        )
        frame["stock_code"] = frame["stock_code"].map(normalize_stock_code)
        frame = frame[frame["stock_code"].isin(codes)].copy()
        frame["source"] = self.name
        frame["timestamp"] = pd.Timestamp.now()
        return frame.reset_index(drop=True)