"""Sohu history provider."""

from __future__ import annotations

import json
from typing import Optional, Sequence, Union
from urllib.request import Request, urlopen

import pandas as pd

from ..symbols import normalize_stock_code, to_number
from .base import BaseProvider


class SohuProvider(BaseProvider):
    name = "sohu"
    daily_adjustments = ("",)
    daily_direct_fields = (
        "date",
        "stock_code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "change",
        "pct_change",
    )

    def fetch_daily(self, code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        return self.fetch_daily_adjusted(code, start_date, end_date, "")

    def fetch_daily_adjusted(self, code: str, start_date: pd.Timestamp, end_date: pd.Timestamp, adjustment: str = "") -> pd.DataFrame:
        adjustment = adjustment or ""
        if adjustment != "":
            raise NotImplementedError("搜狐日线当前仅支持 raw 数据，不提供有效 qfq/hfq 复权。")

        adjust_code = "0"

        symbol = self._to_sohu_symbol(code)
        url = (
            "https://q.stock.sohu.com/hisHq?"
            f"code={symbol}&start={start_date.strftime('%Y%m%d')}&end={end_date.strftime('%Y%m%d')}"
            f"&stat=1&order=D&period=d&rt=json&r=0.0&adjust={adjust_code}"
        )
        payload = self._request_json(url)
        if not payload:
            raise RuntimeError("搜狐日线接口返回空数据。")

        rows = payload[0].get("hq", [])
        if not rows:
            raise RuntimeError("搜狐日线接口返回空数据。")

        parsed = []
        for row in rows:
            if len(row) < 10:
                continue
            parsed.append(
                {
                    "date": row[0],
                    "open": to_number(row[1]),
                    "close": to_number(row[2]),
                    "change": to_number(row[3]),
                    "pct_change": self._to_pct_decimal(row[4]),
                    "low": to_number(row[5]),
                    "high": to_number(row[6]),
                    "volume": to_number(row[7]),
                    "amount": to_number(row[8]),
                    "turnover": to_number(row[9]),
                    "stock_code": code,
                    "source": self.name,
                }
            )
        frame = pd.DataFrame(parsed)
        if frame.empty:
            raise RuntimeError("搜狐日线接口解析后无有效数据。")
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame = frame[(frame["date"] >= start_date.normalize()) & (frame["date"] <= end_date.normalize())]
        if frame.empty:
            raise RuntimeError("搜狐日线接口在指定时间范围内无数据。")
        return frame.sort_values("date").reset_index(drop=True)

    def fetch_realtime(self, codes: Sequence[str]) -> pd.DataFrame:
        rows = []
        end_date = pd.Timestamp.today().normalize()
        start_date = end_date - pd.Timedelta(days=10)
        for code in codes:
            daily = self.fetch_daily(code, start_date, end_date)
            if daily.empty:
                continue
            latest = daily.iloc[-1]
            previous_close = latest.get("close") - latest.get("change") if pd.notna(latest.get("change")) else None
            rows.append(
                {
                    "stock_code": code,
                    "name": latest.get("name"),
                    "price": latest.get("close"),
                    "open": latest.get("open"),
                    "high": latest.get("high"),
                    "low": latest.get("low"),
                    "pre_close": previous_close,
                    "change": latest.get("change"),
                    "pct_change": latest.get("pct_change"),
                    "volume": latest.get("volume"),
                    "amount": latest.get("amount"),
                    "source": self.name,
                    "timestamp": pd.Timestamp.now(),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("搜狐 realtime 快照接口返回空数据。")
        return frame.reset_index(drop=True)

    def fetch_minute(
        self,
        code: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "1m",
        adjust: str = "",
    ) -> pd.DataFrame:
        raise NotImplementedError("搜狐暂未发现稳定可用的结构化分钟接口。")

    @staticmethod
    def _request_json(url: str) -> list[dict]:
        request = Request(url, headers={"Referer": "https://q.stock.sohu.com/", "User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=10) as response:
            text = response.read().decode("utf-8", errors="ignore").strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        raise RuntimeError(f"搜狐接口返回无法解析的内容: {text[:120]}")

    @staticmethod
    def _to_sohu_symbol(code: str) -> str:
        normalized = normalize_stock_code(code)
        if normalized.endswith(".SH"):
            return f"cn_{normalized[:6]}"
        if normalized.endswith(".SZ"):
            return f"cn_{normalized[:6]}"
        if normalized.endswith(".BJ"):
            return f"cn_{normalized[:6]}"
        return f"cn_{normalized[:6]}"

    @staticmethod
    def _to_pct_decimal(value: object) -> float | None:
        numeric = to_number(value)
        if numeric is None:
            return None
        return numeric / 100.0