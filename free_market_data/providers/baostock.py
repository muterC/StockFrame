"""BaoStock provider."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import pandas as pd

from ..symbols import as_timestamp, to_baostock_symbol
from .base import BaseProvider


class BaoStockProvider(BaseProvider):
    name = "baostock"
    daily_adjustments = ("", "qfq", "hfq")

    def fetch_daily(self, code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        return self.fetch_daily_adjusted(code, start_date, end_date, "")

    def fetch_daily_adjusted(self, code: str, start_date: pd.Timestamp, end_date: pd.Timestamp, adjustment: str = "") -> pd.DataFrame:
        import baostock as bs

        adjustflag = {"hfq": "1", "qfq": "2", "": "3", None: "3"}.get(adjustment, "3")
        fields = ",".join(
            [
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
                "turn",
                "pctChg",
                "peTTM",
                "pbMRQ",
                "psTTM",
                "isST",
            ]
        )
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(login.error_msg)
        try:
            query = bs.query_history_k_data_plus(
                to_baostock_symbol(code),
                fields,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag=adjustflag,
            )
            rows = []
            while query.next():
                rows.append(query.get_row_data())
            if not rows:
                return pd.DataFrame()
            raw = pd.DataFrame(rows, columns=query.fields)
        finally:
            bs.logout()

        frame = raw.rename(
            columns={
                "code": "stock_code",
                "preclose": "pre_close",
                "turn": "turnover",
                "pctChg": "pct_change",
                "peTTM": "pe_ttm",
                "pbMRQ": "pb",
                "psTTM": "ps_ttm",
                "isST": "is_st",
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
        import baostock as bs

        frequency = {"5m": "5", "15m": "15", "30m": "30", "60m": "60", "5": "5", "15": "15", "30": "30", "60": "60"}.get(period)
        if frequency is None:
            raise ValueError("BaoStock 分钟线仅支持 5m/15m/30m/60m。")
        start_ts = as_timestamp(start_date) if start_date is not None else pd.Timestamp.today().normalize() - pd.Timedelta(days=5)
        end_ts = as_timestamp(end_date) if end_date is not None else pd.Timestamp.today().normalize()
        adjustflag = {"hfq": "1", "qfq": "2", "": "3", None: "3"}.get(adjust, "3")
        fields = ",".join(["date", "time", "code", "open", "high", "low", "close", "volume", "amount", "adjustflag"])

        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(login.error_msg)
        try:
            query = bs.query_history_k_data_plus(
                to_baostock_symbol(code),
                fields,
                start_date=start_ts.strftime("%Y-%m-%d"),
                end_date=end_ts.strftime("%Y-%m-%d"),
                frequency=frequency,
                adjustflag=adjustflag,
            )
            rows = []
            while query.next():
                rows.append(query.get_row_data())
            if not rows:
                raise RuntimeError("BaoStock 分钟线接口返回空数据。")
            raw = pd.DataFrame(rows, columns=query.fields)
        finally:
            bs.logout()

        frame = raw.rename(columns={"code": "stock_code"})
        time_text = frame["time"].astype(str).str.slice(0, 14)
        frame["datetime"] = pd.to_datetime(time_text, format="%Y%m%d%H%M%S", errors="coerce")
        if frame["datetime"].isna().all():
            frame["datetime"] = pd.to_datetime(frame["date"].astype(str) + " " + frame["time"].astype(str), errors="coerce")
        frame["stock_code"] = code
        frame["source"] = self.name
        return frame.drop(columns=["date", "time", "adjustflag"], errors="ignore").sort_values("datetime").reset_index(drop=True)

    def fetch_realtime(self, codes: Sequence[str]) -> pd.DataFrame:
        today = pd.Timestamp.today().normalize()
        start_date = today - pd.Timedelta(days=15)
        frames = []
        for code in codes:
            daily = self.fetch_daily(code, start_date, today)
            if daily.empty:
                continue
            latest = daily.sort_values("date").tail(1).copy()
            latest["snapshot_date"] = latest["date"]
            latest["price"] = latest["close"]
            latest["timestamp"] = pd.Timestamp.now()
            latest["source"] = self.name
            frames.append(latest.drop(columns=["date"], errors="ignore"))
        if not frames:
            raise RuntimeError("BaoStock 不提供实时行情接口，最新日线快照也为空。")
        return pd.concat(frames, ignore_index=True)