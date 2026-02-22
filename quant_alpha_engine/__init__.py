"""
QuantAlpha_Engine - 工业级因子回测框架
======================================
仿照 WorldQuant 工作流，支持嵌套算子因子生成与严格统计回测。

v2.0 新增：
  - 13 个预构建因子算子（时序 / 量价 / 动量 / 技术指标）
  - fusion 模块：Labeler + StatisticalCombiner + MLCombiner 多因子融合框架

Quick Start（单因子）::

    from quant_alpha_engine import MockDataGenerator
    from quant_alpha_engine.ops import AlphaOps as op
    from quant_alpha_engine.backtest import VectorEngine

    data = MockDataGenerator(n_stocks=100, n_days=504).generate()
    factor = op.Rank(op.Ts_Delta(data.close, 20))
    result = VectorEngine(factor=factor, close=data.close,
                          is_suspended=data.is_suspended, is_limit=data.is_limit).run()
    result.print_summary()
    result.plot()

Quick Start（多因子融合）::

    from quant_alpha_engine import MockDataGenerator
    from quant_alpha_engine.ops import AlphaOps as op
    from quant_alpha_engine.fusion import Labeler, StatisticalCombiner, MLCombiner

    data = MockDataGenerator(n_stocks=100, n_days=504).generate()

    # 构造多个因子
    f1 = op.Rank(op.Ts_Delta(data.close, 20))
    f2 = op.RSI(data.close, window=14)
    f3 = op.RiskAdjMomentum(data.close, window=20, vol_window=10)

    # 生成前向收益率标签
    y = Labeler().set_label(target='close', horizon=5, data={'close': data.close})

    # 统计融合（IC 加权）
    stat = StatisticalCombiner('ic_weighted').fit([f1, f2, f3], y)
    result = stat.evaluate([f1, f2, f3], close=data.close,
                           is_suspended=data.is_suspended, is_limit=data.is_limit,
                           rebalance_freq=5, top_n=30)
    result.print_summary()

    # ML 融合（岭回归，Expanding Window）
    ml = MLCombiner('ridge', min_train_periods=60, refit_freq=20).fit([f1, f2, f3], y)
    result2 = ml.evaluate([f1, f2, f3], close=data.close,
                          is_suspended=data.is_suspended, is_limit=data.is_limit,
                          rebalance_freq=5, top_n=30)
    result2.plot()
"""

from quant_alpha_engine.data.mock_generator import MockDataGenerator, MockData
from quant_alpha_engine.ops.alpha_ops import AlphaOps
from quant_alpha_engine.backtest.vector_engine import VectorEngine, BacktestResult
from quant_alpha_engine.backtest.performance import Performance
from quant_alpha_engine.visualization.report import Report
from quant_alpha_engine.fusion import (
    Labeler,
    FactorCombiner,
    StatisticalCombiner,
    MLCombiner,
)

__version__ = "2.0.0"
__author__  = "QuantAlpha_Engine"

__all__ = [
    # 数据生成
    "MockDataGenerator",
    "MockData",
    # 算子库
    "AlphaOps",
    # 回测引擎
    "VectorEngine",
    "BacktestResult",
    "Performance",
    # 可视化
    "Report",
    # 因子融合（v2.0 新增）
    "Labeler",
    "FactorCombiner",
    "StatisticalCombiner",
    "MLCombiner",
]
