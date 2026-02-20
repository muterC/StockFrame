"""
QuantAlpha_Engine — 完整回测演示
==================================
展示从数据生成 → 算子使用 → 回测 → 报告的完整工作流。

运行方式：
    python demo.py

依赖安装：
    pip install -r requirements.txt
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ============================================================
# 导入框架组件
# ============================================================
from quant_alpha_engine import MockDataGenerator
from quant_alpha_engine.ops import AlphaOps as op
from quant_alpha_engine.backtest import VectorEngine


# ============================================================
# 1. 生成模拟市场数据
# ============================================================
print("=" * 60)
print("  QuantAlpha_Engine — 因子回测框架演示")
print("=" * 60)
print("\n[Step 1] 生成模拟市场数据（100只股票 × 504个交易日）...")

gen  = MockDataGenerator(n_stocks=100, n_days=504, n_industries=10, seed=42)
data = gen.generate()

close    = data.close
volume   = data.volume
industry = data.industry
is_susp  = data.is_suspended
is_limit = data.is_limit

print(f"  ✓ 价格数据维度：{close.shape}  (日期 × 股票)")
print(f"  ✓ 停牌天数比例：{is_susp.values.mean()*100:.2f}%")
print(f"  ✓ 涨跌停天数比例：{is_limit.values.mean()*100:.2f}%")


# ============================================================
# 2. 构建因子（演示嵌套算子）
# ============================================================
print("\n[Step 2] 构建 Alpha 因子...")

# --- 因子 1：反转动量因子 ---
# 逻辑：短期超跌的股票均值回归预期更强
# Rank(-Ts_Delta(close, 5))  => 近5日跌幅最大的股票排名靠前
factor_reversal = op.Rank(-op.Ts_Delta(close, 5))
print("  ✓ 因子1 (反转动量): Rank(-Ts_Delta(close, 5))")

# --- 因子 2：量价相关因子（行业中性化）---
# 逻辑：成交量与价格负相关（缩量上涨）为强势信号
# Neutralize(Rank(-Ts_Corr(volume, close, 10)), industry)
raw_corr_factor = op.Ts_Corr(volume, close, window=10)
factor_volprice = op.Neutralize(op.Rank(-raw_corr_factor), industry)
print("  ✓ 因子2 (量价因子): Neutralize(Rank(-Ts_Corr(volume, close, 10)), industry)")

# --- 因子 3：综合技术因子 ---
# 逻辑：结合短期动量、价格位置和波动率
# ZScore(Decay_Linear(Rank(Ts_Delta(close,10)), 5))
factor_tech = op.ZScore(
    op.Decay_Linear(
        op.Rank(op.Ts_Delta(close, 10)),
        d=5
    )
)
print("  ✓ 因子3 (技术因子): ZScore(Decay_Linear(Rank(Ts_Delta(close, 10)), 5))")


# ============================================================
# 3. 分别回测三个因子
# ============================================================
print("\n[Step 3] 执行因子回测...\n")

factors = {
    "反转动量因子": factor_reversal,
    "量价相关因子": factor_volprice,
    "综合技术因子": factor_tech,
}

results = {}

for name, factor in factors.items():
    print(f"{'─'*50}")
    print(f"  正在回测：{name}")
    print(f"{'─'*50}")

    engine = VectorEngine(
        factor         = factor,
        close          = close,
        is_suspended   = is_susp,
        is_limit       = is_limit,
        rebalance_freq = 5,       # 每周调仓
        top_n          = 30,      # 持仓 30 只
        weight_method  = "equal", # 等权
        cost_rate      = 0.0015,  # 单边 0.15%
    )
    result = engine.run()
    results[name] = result

    # 打印绩效摘要
    result.print_summary()


# ============================================================
# 4. 选择最优因子（按 Sharpe Ratio）并绘制完整报告
# ============================================================
print("\n[Step 4] 选择最优因子，生成回测分析报告...\n")

best_name = max(results, key=lambda k: results[k].metrics.get("Sharpe_Ratio", -999))
best_result = results[best_name]

print(f"  🏆 最优因子：{best_name}")
print(f"     Sharpe = {best_result.metrics['Sharpe_Ratio']:.4f}")
print(f"     ICIR   = {best_result.metrics['ICIR']:.4f}")
print(f"     Fitness = {best_result.metrics['Fitness']:.4f}")

print("\n  正在生成 Matplotlib 可视化报告（6 子图）...")
best_result.plot()  # 弹窗展示

# 若需保存到文件，取消下方注释：
# best_result.plot(save_path="backtest_report.png")


# ============================================================
# 5. 高级用法：自定义组合因子回测
# ============================================================
print("\n[Step 5] 高级示例：自定义因子权重组合回测")
print("─" * 50)

# 构建自定义复合因子（多因子线性加权）
# factor_alpha = 0.5 * Rank(factor1) + 0.5 * Rank(factor2)
alpha_combo = (
    0.5 * op.Rank(factor_reversal) +
    0.5 * op.Rank(factor_volprice)
)

print("  因子：0.5 × Rank(factor_reversal) + 0.5 × Rank(factor_volprice)")

# 因子加权持仓（非等权，按因子绝对值分配）
engine_combo = VectorEngine(
    factor         = alpha_combo,
    close          = close,
    is_suspended   = is_susp,
    is_limit       = is_limit,
    rebalance_freq = 10,              # 每 2 周调仓
    top_n          = 20,              # 精选 20 只
    weight_method  = "factor_weighted",  # 因子值加权
    cost_rate      = 0.0015,
)
result_combo = engine_combo.run()
result_combo.print_summary()

print("\n[完成] QuantAlpha_Engine 演示结束。")
print("使用说明：")
print("  1. 将 close/volume 等价格数据替换为您的真实数据")
print("  2. 用 AlphaOps 算子组合构造因子")
print("  3. 调用 VectorEngine(...).run() 得到 BacktestResult")
print("  4. result.print_summary() 查看指标，result.plot() 生成图表")
