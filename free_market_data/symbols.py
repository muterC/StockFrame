"""Symbol and value normalization helpers."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import pandas as pd


def normalize_stock_code(code: str) -> str:
    """Normalize common A-share/Yahoo symbols to forms like 600000.SH."""
    clean = str(code).strip().upper().replace("_", ".")
    if not clean:
        raise ValueError("股票代码不能为空。")
    if "." in clean:
        left, right = clean.split(".", 1)
        if left.isalpha() and right.isdigit():
            return f"{right}.{left}"
        return f"{left}.{right}"
    if len(clean) == 8 and clean[:2] in {"SH", "SZ", "BJ"} and clean[2:].isdigit():
        return f"{clean[2:]}.{clean[:2]}"
    if len(clean) == 8 and clean[-2:] in {"SH", "SZ", "BJ"} and clean[:6].isdigit():
        return f"{clean[:6]}.{clean[-2:]}"
    if len(clean) == 6 and clean.isdigit():
        if clean.startswith(("5", "6", "9")):
            return f"{clean}.SH"
        if clean.startswith(("0", "2", "3")):
            return f"{clean}.SZ"
        if clean.startswith(("4", "8")):
            return f"{clean}.BJ"
    return clean


def normalize_codes(stock_codes: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(stock_codes, str):
        stock_codes = [stock_codes]
    return [normalize_stock_code(code) for code in stock_codes]


def as_timestamp(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    return pd.to_datetime(value).normalize()


def is_a_share(code: str) -> bool:
    return len(code) == 9 and code[:6].isdigit() and code[-3:] in {".SH", ".SZ", ".BJ"}


def to_akshare_symbol(code: str) -> str:
    if not is_a_share(code):
        raise ValueError(f"AkShare A 股接口不支持该代码: {code}")
    return code[:6]


def to_baostock_symbol(code: str) -> str:
    if not is_a_share(code) or code.endswith(".BJ"):
        raise ValueError(f"Baostock 当前默认实现不支持该代码: {code}")
    return f"{code[-2:].lower()}.{code[:6]}"


def to_yahoo_symbol(code: str) -> str:
    if code.endswith(".SH"):
        return f"{code[:6]}.SS"
    if code.endswith(".SZ"):
        return code
    return code


def to_tencent_symbol(code: str) -> str:
    if code.endswith(".SH"):
        return f"sh{code[:6]}"
    if code.endswith(".SZ"):
        return f"sz{code[:6]}"
    if code.endswith(".BJ"):
        return f"bj{code[:6]}"
    return code.lower()


def to_xueqiu_symbol(code: str) -> str:
    if not is_a_share(code):
        raise ValueError(f"雪球 A 股接口不支持该代码: {code}")
    return f"{code[-2:]}{code[:6]}"


def to_number(value: object) -> Optional[float]:
    try:
        if value in {None, "", "--"}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None