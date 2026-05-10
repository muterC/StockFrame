"""Tencent quote provider."""

from __future__ import annotations

import json
from typing import Optional, Sequence, Union
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd

from ..symbols import as_timestamp, normalize_stock_code, to_number, to_tencent_symbol
from .base import BaseProvider


class TencentProvider(BaseProvider):
    name = "tencent"
    daily_adjustments = ("", "qfq", "hfq")
    daily_direct_fields = ("date", "stock_code", "open", "high", "low", "close", "volume", "amount")

    def fetch_daily(self, code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        return self.fetch_daily_adjusted(code, start_date, end_date, "")

    def fetch_daily_adjusted(self, code: str, start_date: pd.Timestamp, end_date: pd.Timestamp, adjustment: str = "") -> pd.DataFrame:
        symbol = to_tencent_symbol(code)
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")
        count = max((end_date.normalize() - start_date.normalize()).days + 10, 10)
        if adjustment in {"qfq", "hfq"}:
            path = "fqkline/get"
            param = f"{symbol},day,{start},{end},{count},{adjustment}"
            key = f"{adjustment}day"
        else:
            path = "kline/kline"
            param = f"{symbol},day,{start},{end},{count}"
            key = "day"
        payload = self._fetch_json(f"https://web.ifzq.gtimg.cn/appstock/app/{path}?param={quote(param, safe=',')}")
        rows = payload.get("data", {}).get(symbol, {}).get(key) or payload.get("data", {}).get(symbol, {}).get("day") or []
        frame = self._kline_rows_to_frame(rows, code, daily=True)
        if frame.empty:
            raise RuntimeError("腾讯日线接口返回空数据。")
        frame = frame[(frame["date"] >= start_date.normalize()) & (frame["date"] <= end_date.normalize())]
        if frame.empty:
            raise RuntimeError("腾讯日线接口在指定时间范围内无数据。")
        return frame.reset_index(drop=True)

    def fetch_minute(
        self,
        code: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "1m",
        adjust: str = "",
    ) -> pd.DataFrame:
        symbol = to_tencent_symbol(code)
        period_key = self._minute_period(period)
        payload = self._fetch_json(f"https://web.ifzq.gtimg.cn/appstock/app/kline/mkline?param={quote(f'{symbol},{period_key},,320', safe=',')}")
        rows = payload.get("data", {}).get(symbol, {}).get(period_key) or []
        frame = self._kline_rows_to_frame(rows, code, daily=False)
        if frame.empty:
            raise RuntimeError("腾讯分钟线接口返回空数据。")
        if start_date is not None:
            frame = frame[frame["datetime"] >= as_timestamp(start_date)]
        if end_date is not None:
            frame = frame[frame["datetime"] <= as_timestamp(end_date)]
        if frame.empty:
            raise RuntimeError("腾讯分钟线接口在指定时间范围内无数据。")
        return frame.reset_index(drop=True)

    def fetch_realtime(self, codes: Sequence[str]) -> pd.DataFrame:
        query_codes = [to_tencent_symbol(code) for code in codes]
        url = "http://qt.gtimg.cn/q=" + quote(",".join(query_codes))
        with urlopen(url, timeout=10) as response:
            text = response.read().decode("gbk", errors="ignore")

        rows = []
        for quote_text in text.split(";"):
            if "=\"" not in quote_text:
                continue
            payload = quote_text.split("=\"", 1)[1].strip().strip('"')
            parts = payload.split("~")
            if len(parts) < 35:
                continue
            rows.append(
                {
                    "stock_code": normalize_stock_code(parts[2]),
                    "name": parts[1],
                    "price": to_number(parts[3]),
                    "pre_close": to_number(parts[4]),
                    "open": to_number(parts[5]),
                    "change": to_number(parts[31]),
                    "pct_change": to_number(parts[32]),
                    "high": to_number(parts[33]),
                    "low": to_number(parts[34]),
                    "source": self.name,
                    "timestamp": pd.Timestamp.now(),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("腾讯实时行情接口返回空数据。")
        frame = frame[frame["stock_code"].isin(codes)]
        if frame.empty:
            raise RuntimeError("腾讯实时行情接口未返回请求的股票代码。")
        return frame.reset_index(drop=True)

    @staticmethod
    def _fetch_json(url: str) -> dict:
        with urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))

    @staticmethod
    def _minute_period(period: str) -> str:
        mapping = {"1m": "m1", "5m": "m5", "15m": "m15", "30m": "m30", "60m": "m60", "1": "m1", "5": "m5", "15": "m15", "30": "m30", "60": "m60"}
        if period not in mapping:
            raise ValueError("腾讯分钟线仅支持 1m/5m/15m/30m/60m。")
        return mapping[period]

    def _kline_rows_to_frame(self, rows: Sequence[Sequence[object]], code: str, daily: bool) -> pd.DataFrame:
        parsed = []
        for row in rows:
            if len(row) < 6:
                continue
            parsed.append(
                {
                    "date" if daily else "datetime": row[0],
                    "open": to_number(row[1]),
                    "close": to_number(row[2]),
                    "high": to_number(row[3]),
                    "low": to_number(row[4]),
                    "volume": to_number(row[5]),
                    "amount": to_number(row[6]) if len(row) > 6 else None,
                    "stock_code": code,
                    "source": self.name,
                }
            )
        frame = pd.DataFrame(parsed)
        if frame.empty:
            return frame
        if daily:
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            return frame.sort_values("date").reset_index(drop=True)
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        return frame.sort_values("datetime").reset_index(drop=True)