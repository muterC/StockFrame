"""
QuantAlpha_Engine - 工业级因子回测框架
======================================
仿照 WorldQuant 工作流，支持嵌套算子因子生成与严格统计回测。

Quick Start:
    from quant_alpha_engine import MockDataGenerator
    from quant_alpha_engine.ops import AlphaOps as op
    from quant_alpha_engine.backtest import VectorEngine

    data = MockDataGenerator(n_stocks=100, n_days=504).generate()
    factor = op.Rank(op.Ts_Delta(data.close, 20))
    result = VectorEngine(factor=factor, close=data.close,
                          is_suspended=data.is_suspended, is_limit=data.is_limit).run()
    result.print_summary()
    result.plot()
"""

from quant_alpha_engine.data.mock_generator import MockDataGenerator, MockData
from quant_alpha_engine.ops.alpha_ops import AlphaOps
from quant_alpha_engine.backtest.vector_engine import VectorEngine, BacktestResult
from quant_alpha_engine.backtest.performance import Performance
from quant_alpha_engine.visualization.report import Report

__version__ = "1.0.0"
__author__  = "QuantAlpha_Engine"

__all__ = [
    "MockDataGenerator",
    "MockData",
    "AlphaOps",
    "VectorEngine",
    "BacktestResult",
    "Performance",
    "Report",
]
