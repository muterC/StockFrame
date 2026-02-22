"""
quant_alpha_engine/fusion/__init__.py
======================================
因子融合子包 — 多因子合成框架

导出类
------
Labeler             : 前向收益率标签生成器
FactorCombiner      : 因子融合抽象基类（ABC）
StatisticalCombiner : 统计融合（等权 / IC加权 / 最小方差）
MLCombiner          : 机器学习融合（Expanding Window，防未来函数）

Quick Start::

    from quant_alpha_engine.fusion import (
        Labeler,
        StatisticalCombiner,
        MLCombiner,
    )

    # 生成标签
    y = Labeler().set_label(target='close', horizon=5, data={'close': close_df})

    # 统计融合
    stat = StatisticalCombiner('ic_weighted').fit([f1, f2, f3], y)
    result = stat.evaluate([f1, f2, f3], close=close, is_suspended=susp,
                           is_limit=limit, rebalance_freq=5, top_n=30)
    result.print_summary()

    # ML 融合
    ml = MLCombiner('ridge', min_train_periods=60, refit_freq=20)
    ml.fit([f1, f2, f3], y)
    result2 = ml.evaluate([f1, f2, f3], close=close, is_suspended=susp,
                          is_limit=limit, rebalance_freq=5, top_n=30)
    result2.plot()
"""

from quant_alpha_engine.fusion.labeler import Labeler
from quant_alpha_engine.fusion.combiner import (
    FactorCombiner,
    StatisticalCombiner,
    MLCombiner,
)

__all__ = [
    "Labeler",
    "FactorCombiner",
    "StatisticalCombiner",
    "MLCombiner",
]
