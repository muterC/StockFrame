"""Provider registry for the standalone free market data store."""

from __future__ import annotations

from .akshare import AkShareProvider
from .baostock import BaoStockProvider
from .base import BaseProvider
from .provider import Provider
from .tencent import TencentProvider
from .xueqiu import XueqiuProvider
from .yahoo import YahooProvider

DEFAULT_PROVIDER_CLASSES = {
    "akshare": AkShareProvider,
    "baostock": BaoStockProvider,
    "yahoo": YahooProvider,
    "tencent": TencentProvider,
    "xueqiu": XueqiuProvider,
}

__all__ = ["BaseProvider", "Provider", "DEFAULT_PROVIDER_CLASSES"]