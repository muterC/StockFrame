"""
data_loader
===========
从 DataRepository 加载真实 A 股市场数据，输出与 MockData 完全兼容的
MarketData 容器，可直接替换 MockDataGenerator 用于 AlphaOps + VectorEngine。

Quick Start::

    from data_loader import RepositoryDataLoader

    loader = RepositoryDataLoader("rawdata/repository")
    data   = loader.load(
        symbols    = ['600519', '000001', '000002'],
        start_date = '2024-01-01',
        end_date   = '2026-02-22',
        adj_type   = 'hfq',
    )
    data.print_summary()

    # 与 MockData 完全相同的用法
    from quant_alpha_engine.ops import AlphaOps as op
    from quant_alpha_engine.backtest import VectorEngine

    factor = op.Rank(op.Ts_Delta(data.close, 20))
    result = VectorEngine(
        factor       = factor,
        close        = data.close,
        is_suspended = data.is_suspended,
        is_limit     = data.is_limit,
    ).run()
    result.print_summary()
"""

from data_loader.loader import MarketData, RepositoryDataLoader

__version__ = "1.0.0"

__all__ = [
    "RepositoryDataLoader",
    "MarketData",
]
