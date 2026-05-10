"""Yahoo Finance provider."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import pandas as pd

from ..symbols import as_timestamp, to_yahoo_symbol
from .base import BaseProvider


class YahooProvider(BaseProvider):
    name = "yahoo"

    def fetch_daily(self, code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            to_yahoo_symbol(code),
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frame = raw.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
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
        import yfinance as yf

        interval = period if period.endswith("m") else f"{period}m"
        raw = yf.Ticker(to_yahoo_symbol(code)).history(
            interval=interval,
            start=as_timestamp(start_date).strftime("%Y-%m-%d") if start_date is not None else None,
            end=as_timestamp(end_date).strftime("%Y-%m-%d") if end_date is not None else None,
            auto_adjust=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        frame = raw.reset_index().rename(
            columns={
                "Datetime": "datetime",
                "Date": "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame["stock_code"] = code
        frame["source"] = self.name
        return frame.sort_values("datetime").reset_index(drop=True)

    def fetch_realtime(self, codes: Sequence[str]) -> pd.DataFrame:
        import yfinance as yf

        rows = []
        for code in codes:
            ticker = yf.Ticker(to_yahoo_symbol(code))
            info = getattr(ticker, "fast_info", {}) or {}
            rows.append(
                {
                    "stock_code": code,
                    "price": info.get("last_price"),
                    "open": info.get("open"),
                    "high": info.get("day_high"),
                    "low": info.get("day_low"),
                    "pre_close": info.get("previous_close"),
                    "volume": info.get("last_volume"),
                    "source": self.name,
                    "timestamp": pd.Timestamp.now(),
                }
            )
        return pd.DataFrame(rows).dropna(how="all", subset=["price", "open", "high", "low", "pre_close"])