"""
quant_alpha_engine/tuning/__init__.py
======================================
参数调优子包 — 因子 + 回测参数的笛卡尔积搜索框架

导出类
------
ParameterTuner : 参数调优引擎

Quick Start::

    from quant_alpha_engine.tuning import ParameterTuner
    from quant_alpha_engine.ops import AlphaOps as op

    # 定义因子工厂函数
    def my_factor(close, volume, high, low, window=20, vol_window=10):
        return op.Rank(op.RiskAdjMomentum(close, window=window, vol_window=vol_window))

    # 创建调优器
    tuner = ParameterTuner(
        factor_fn       = my_factor,
        close           = close,
        is_suspended    = is_susp,
        is_limit        = is_limit,
        factor_params   = {'window': [10, 20, 30], 'vol_window': [5, 10]},
        backtest_params = {'rebalance_freq': [1, 5], 'top_n': [20, 30]},
        opt_target      = {'Sharpe_Ratio': 'desc', '最大回撤': 'asc'},
        n_jobs          = 4,
    )

    tuner.run()
    tuner.print_top(5)           # Jupyter 显示彩色 HTML 表格
    tuner.get_result(0).plot()   # 最优组合详细图表
"""

from quant_alpha_engine.tuning.tuner import ParameterTuner

__all__ = ["ParameterTuner"]
