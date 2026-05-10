"""Base provider interface used by the unified Provider facade."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import pandas as pd

FieldSelection = Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]]

PROTECTED_COLUMNS = {
    "daily": (
        "date",
        "stock_code",
        "open",
        "high",
        "low",
        "close",
        "qfq_factor",
        "hfq_factor",
        "price_source",
        "source",
        "updated_at",
    ),
    "minute": ("datetime", "stock_code", "source"),
    "realtime": ("stock_code", "source", "timestamp"),
}


class BaseProvider:
    """Provider interface for free market data sources."""

    name = "base"
    daily_adjustments: Sequence[str] = ("",)
    daily_direct_fields: Sequence[str] = ()

    def fetch_daily(
        self,
        code: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} 未实现日线接口。")

    def fetch_daily_adjusted(
        self,
        code: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        adjustment: str = "",
    ) -> pd.DataFrame:
        adjustment = adjustment or ""
        if adjustment == "":
            return self.fetch_daily(code, start_date, end_date)
        raise NotImplementedError(f"{self.name} 不支持 {adjustment} 日线。")

    def fetch_minute(
        self,
        code: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "1m",
        adjust: str = "",
    ) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} 未实现分钟级接口。")

    def fetch_realtime(self, codes: Sequence[str]) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} 未实现实时行情接口。")

    @staticmethod
    def empty() -> pd.DataFrame:
        return pd.DataFrame()