"""Sina quote provider."""

from __future__ import annotations

import json
from typing import Optional, Sequence, Union
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from ..symbols import as_timestamp, normalize_stock_code, to_number, to_tencent_symbol
from .base import BaseProvider


class SinaProvider(BaseProvider):
    name = "sina"

    def fetch_daily(self, code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError("新浪日线接口缺少可用复权因子，当前不作为 daily provider 使用。")

    def fetch_minute(
        self,
        code: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "1m",
        adjust: str = "",
    ) -> pd.DataFrame:
        scale = self._minute_scale(period)
        datalen = self._minute_datalen(start_date, end_date, scale)
        frame = self._fetch_kline(code, scale=scale, datalen=datalen)
        if frame.empty:
            raise RuntimeError("新浪分钟接口返回空数据。")
        if start_date is not None:
            frame = frame[frame["datetime"] >= as_timestamp(start_date)]
        if end_date is not None:
            frame = frame[frame["datetime"] <= as_timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
        if frame.empty:
            raise RuntimeError("新浪分钟接口在指定时间范围内无数据。")
        frame["stock_code"] = code
        frame["source"] = self.name
        return frame.sort_values("datetime").reset_index(drop=True)

    def fetch_realtime(self, codes: Sequence[str]) -> pd.DataFrame:
        query_codes = [self._to_sina_symbol(code) for code in codes]
        request = Request(
            "https://hq.sinajs.cn/list=" + quote(",".join(query_codes)),
            headers={"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"},
        )
        with urlopen(request, timeout=10) as response:
            text = response.read().decode("gbk", errors="ignore")

        rows = []
        for quote_text in text.split(";"):
            if "=\"" not in quote_text:
                continue
            prefix, payload = quote_text.split("=\"", 1)
            payload = payload.strip().strip('"')
            if not payload:
                continue
            parts = payload.split(",")
            if len(parts) < 10:
                continue
            raw_symbol = prefix.rsplit("_", 1)[-1]
            rows.append(
                {
                    "stock_code": normalize_stock_code(raw_symbol[2:] + "." + raw_symbol[:2].upper()),
                    "name": parts[0],
                    "open": to_number(parts[1]),
                    "pre_close": to_number(parts[2]),
                    "price": to_number(parts[3]),
                    "high": to_number(parts[4]),
                    "low": to_number(parts[5]),
                    "volume": to_number(parts[8]),
                    "amount": to_number(parts[9]),
                    "bid": to_number(parts[6]),
                    "ask": to_number(parts[7]),
                    "change": self._safe_change(parts[3], parts[2]),
                    "pct_change": self._safe_pct_change(parts[3], parts[2]),
                    "source": self.name,
                    "timestamp": pd.Timestamp.now(),
                }
            )

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("新浪实时行情接口返回空数据。")
        frame = frame[frame["stock_code"].isin(codes)]
        if frame.empty:
            raise RuntimeError("新浪实时行情接口未返回请求的股票代码。")
        return frame.reset_index(drop=True)

    def _fetch_kline(self, code: str, scale: int, datalen: int) -> pd.DataFrame:
        symbol = self._to_sina_symbol(code)
        request = Request(
            "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?"
            f"symbol={quote(symbol)}&scale={scale}&ma=no&datalen={datalen}",
            headers={"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"},
        )
        with urlopen(request, timeout=10) as response:
            text = response.read().decode("utf-8", errors="ignore").strip()
        if not text:
            return pd.DataFrame()
        payload = json.loads(text)
        frame = pd.DataFrame(payload)
        if frame.empty:
            return frame
        date_column = "day" if "day" in frame.columns else "date"
        frame = frame.rename(columns={date_column: "datetime"})
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        for column in ("open", "high", "low", "close", "volume"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["source"] = self.name
        return frame

    @staticmethod
    def _minute_scale(period: str) -> int:
        mapping = {"1m": 1, "1": 1, "5m": 5, "5": 5, "15m": 15, "15": 15, "30m": 30, "30": 30, "60m": 60, "60": 60}
        if period not in mapping:
            raise ValueError("新浪分钟接口仅支持 1m/5m/15m/30m/60m。")
        return mapping[period]

    @staticmethod
    def _minute_datalen(start_date: Optional[Union[str, pd.Timestamp]], end_date: Optional[Union[str, pd.Timestamp]], scale: int) -> int:
        if start_date is None or end_date is None:
            return 320
        start_ts = as_timestamp(start_date)
        end_ts = as_timestamp(end_date)
        days = max((end_ts - start_ts).days + 1, 1)
        estimated = int(days * (240 / max(scale, 1))) + 20
        return max(estimated, 20)

    @staticmethod
    def _to_sina_symbol(code: str) -> str:
        symbol = to_tencent_symbol(code)
        if symbol.startswith(("sh", "sz", "bj")):
            return symbol
        return code.lower()

    @staticmethod
    def _safe_change(price: object, pre_close: object) -> float | None:
        price_value = to_number(price)
        pre_close_value = to_number(pre_close)
        if price_value is None or pre_close_value is None:
            return None
        return price_value - pre_close_value

    @staticmethod
    def _safe_pct_change(price: object, pre_close: object) -> float | None:
        price_value = to_number(price)
        pre_close_value = to_number(pre_close)
        if price_value is None or pre_close_value in {None, 0}:
            return None
        return (price_value / pre_close_value) - 1.0