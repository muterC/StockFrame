"""Xueqiu provider."""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence, Union
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from ..symbols import as_timestamp, normalize_stock_code, to_xueqiu_symbol
from .base import BaseProvider


class XueqiuProvider(BaseProvider):
    name = "xueqiu"
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
        "volume_post",
        "amount_post",
        "change",
        "pct_change",
        "turnover",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "pcf_ttm",
        "total_market_cap",
    )

    def __init__(self, cookie: Optional[str] = None, user_agent: Optional[str] = None) -> None:
        self.cookie = cookie or os.getenv("XUEQIU_COOKIE")
        self.user_agent = user_agent or os.getenv(
            "XUEQIU_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        )

    def fetch_daily(self, code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        return self.fetch_daily_adjusted(code, start_date, end_date, "")

    def fetch_daily_adjusted(self, code: str, start_date: pd.Timestamp, end_date: pd.Timestamp, adjustment: str = "") -> pd.DataFrame:
        symbol = to_xueqiu_symbol(code)
        count = max((end_date.normalize() - start_date.normalize()).days * 2 + 10, 10)
        query = urlencode(
            {
                "symbol": symbol,
                "begin": int(end_date.timestamp() * 1000),
                "period": "day",
                "type": {"": "normal", None: "normal", "qfq": "before", "hfq": "after"}.get(adjustment, "normal"),
                "count": -count,
                "indicator": "kline,pe,pb,ps,pcf,market_capital",
            }
        )
        payload = self._request_json(f"https://stock.xueqiu.com/v5/stock/chart/kline.json?{query}")
        data = payload.get("data", {})
        columns = data.get("column", [])
        rows = data.get("item", [])
        if not columns or not rows:
            raise RuntimeError("雪球日线接口返回空数据。")
        frame = pd.DataFrame(rows, columns=columns).rename(
            columns={
                "timestamp": "date",
                "chg": "change",
                "percent": "pct_change",
                "turnoverrate": "turnover",
                "pe": "pe_ttm",
                "ps": "ps_ttm",
                "pcf": "pcf_ttm",
                "market_capital": "total_market_cap",
            }
        )
        frame["date"] = pd.to_datetime(frame["date"], unit="ms").dt.normalize()
        frame["stock_code"] = code
        frame = frame[(frame["date"] >= start_date.normalize()) & (frame["date"] <= end_date.normalize())]
        if frame.empty:
            raise RuntimeError("雪球日线接口在指定时间范围内无数据。")
        return frame.reset_index(drop=True)

    def fetch_minute(
        self,
        code: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "1m",
        adjust: str = "",
    ) -> pd.DataFrame:
        if period not in {"1m", "1"}:
            raise ValueError("雪球分钟接口当前仅支持 1m。")
        symbol = to_xueqiu_symbol(code)
        payload = self._request_json(f"https://stock.xueqiu.com/v5/stock/chart/minute.json?{urlencode({'symbol': symbol, 'period': '1d'})}")
        rows = payload.get("data", {}).get("items", [])
        if not rows:
            raise RuntimeError("雪球分钟接口返回空数据。")
        frame = pd.DataFrame(rows).rename(
            columns={
                "timestamp": "datetime",
                "current": "close",
                "avg_price": "vwap",
                "chg": "change",
                "percent": "pct_change",
            }
        )
        frame["datetime"] = pd.to_datetime(frame["datetime"], unit="ms")
        frame["stock_code"] = code
        frame["source"] = self.name
        if start_date is not None:
            frame = frame[frame["datetime"] >= as_timestamp(start_date)]
        if end_date is not None:
            frame = frame[frame["datetime"] <= as_timestamp(end_date)]
        if frame.empty:
            raise RuntimeError("雪球分钟接口在指定时间范围内无数据。")
        return frame.sort_values("datetime").reset_index(drop=True)

    def fetch_realtime(self, codes: Sequence[str]) -> pd.DataFrame:
        symbols = [to_xueqiu_symbol(code) for code in codes]
        payload = self._request_json(f"https://stock.xueqiu.com/v5/stock/realtime/quotec.json?{urlencode({'symbol': ','.join(symbols)})}")
        rows = payload.get("data", [])
        if not rows:
            raise RuntimeError("雪球实时行情接口返回空数据。")
        frame = pd.DataFrame(rows).rename(
            columns={
                "symbol": "stock_code",
                "current": "price",
                "last_close": "pre_close",
                "chg": "change",
                "percent": "pct_change",
                "turnover_rate": "turnover",
                "market_capital": "total_market_cap",
                "float_market_capital": "float_market_cap",
            }
        )
        frame["stock_code"] = frame["stock_code"].map(normalize_stock_code)
        frame["source"] = self.name
        frame["timestamp"] = pd.Timestamp.now()
        frame = frame[frame["stock_code"].isin(codes)]
        if frame.empty:
            raise RuntimeError("雪球实时行情接口未返回请求的股票代码。")
        return frame.reset_index(drop=True)

    def _request_json(self, url: str) -> dict:
        if not self.cookie:
            raise RuntimeError("雪球接口需要登录态。请设置环境变量 XUEQIU_COOKIE，或创建 XueqiuProvider(cookie='...')。")
        request = Request(
            url,
            headers={
                "Cookie": self.cookie,
                "User-Agent": self.user_agent,
                "Referer": "https://xueqiu.com/",
            },
        )
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))