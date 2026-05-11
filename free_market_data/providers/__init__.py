"""Provider registry for the standalone free market data store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .akshare import AkShareProvider
from .baostock import BaoStockProvider
from .base import BaseProvider
from .hyper_provider import HyperProvider
from .sina import SinaProvider
from .sohu import SohuProvider
from .tencent import TencentProvider
from .xueqiu import XueqiuProvider
from .yahoo import YahooProvider

if TYPE_CHECKING:
    pass

DEFAULT_PROVIDER_CLASSES = {
    "akshare": AkShareProvider,
    "baostock": BaoStockProvider,
    "yahoo": YahooProvider,
    "tencent": TencentProvider,
    "xueqiu": XueqiuProvider,
    "sina": SinaProvider,
    "sohu": SohuProvider,
}

__all__ = ["BaseProvider", "HyperProvider", "DEFAULT_PROVIDER_CLASSES"]