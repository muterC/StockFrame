# QuantAlpha Engine

> 工业级因子研发与回测框架 · 仿照 WorldQuant 工作流

**v2.0.0 新特性** 🚀 新增 13 个预构建算子（时序/量价/动量/技术指标）+ `fusion` 多因子融合模块（统计融合 / 机器学习融合），支持一键输出 `BacktestResult`。

---

## 目录

- [项目简介](#项目简介)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [依赖安装](#依赖安装)
- [模块详解](#模块详解)
  - [MockDataGenerator — 模拟数据生成器](#1-mockdatagenerator--模拟数据生成器)
  - [AlphaOps — 算子库](#2-alphaops--算子库)
    - [时序类算子 (v1)](#时序类-time-series)
    - [截面类算子 (v1)](#截面类-cross-sectional)
    - [特殊类算子 (v1)](#特殊类-special)
    - [🆕 时序统计算子 (v2)](#-时序统计算子-v2)
    - [🆕 量价因子算子 (v2)](#-量价因子算子-v2)
    - [🆕 动量因子算子 (v2)](#-动量因子算子-v2)
    - [🆕 技术指标算子 (v2)](#-技术指标算子-v2)
  - [VectorEngine — 回测引擎](#3-vectorengine--回测引擎)
  - [BacktestResult — 回测结果对象](#4-backtestresult--回测结果对象)
  - [Performance — 绩效指标计算](#5-performance--绩效指标计算)
  - [Report — 可视化报告](#6-report--可视化报告)
  - [🆕 Fusion — 多因子融合框架 (v2)](#7-fusion--多因子融合框架-v2)
    - [Labeler — 标签生成器](#71-labeler--标签生成器)
    - [StatisticalCombiner — 统计融合](#72-statisticalcombiner--统计融合)
    - [MLCombiner — 机器学习融合](#73-mlcombiner--机器学习融合)
- [指标说明](#指标说明)
- [使用真实数据](#使用真实数据)
- [因子构造示例集](#因子构造示例集)
- [常见问题](#常见问题)

---

## 项目简介

QuantAlpha Engine 是一个面向量化研究员的**因子回测一站式框架**，核心设计理念仿照 WorldQuant Alpha 平台工作流：

- 用**嵌套算子**描述因子逻辑，逻辑清晰，可读性强
- **全向量化**计算，所有操作基于 pandas/numpy 矩阵运算，无显式股票循环
- 严格防止**未来函数（Look-ahead bias）**：T 日因子值 → T+1 日实际收益
- 支持**停牌、涨跌停**自动过滤，扣除真实交易成本
- 一行代码 `.run()` 得到完整绩效报告

**最简使用流程：**

```python
from quant_alpha_engine import MockDataGenerator
from quant_alpha_engine.ops import AlphaOps as op
from quant_alpha_engine.backtest import VectorEngine

# 1. 准备数据
data = MockDataGenerator(n_stocks=100, n_days=504).generate()

# 2. 用算子构造因子（支持任意嵌套）
factor = op.Rank(op.Ts_Delta(data.close, 20))

# 3. 一行调用回测
result = VectorEngine(
    factor=factor, close=data.close,
    is_suspended=data.is_suspended, is_limit=data.is_limit
).run()

# 4. 查看结果
result.print_summary()   # 控制台打印指标表格
result.plot()            # 生成 6 子图分析报告
```

---

## 目录结构

```
quant_alpha_engine/
├── __init__.py                    # 统一导出，一行导入所有模块（v2.0.0）
├── data/
│   └── mock_generator.py          # 模拟数据生成器
├── ops/
│   └── alpha_ops.py               # 算子库（v1: 13个 + v2: 13个新算子）
├── backtest/
│   ├── performance.py             # 绩效指标计算
│   └── vector_engine.py           # 矩阵式净值回测引擎
├── fusion/                        # 🆕 v2.0 多因子融合框架
│   ├── __init__.py                # 导出 Labeler / StatisticalCombiner / MLCombiner
│   ├── labeler.py                 # 前向收益率标签生成器
│   └── combiner.py                # 统计融合 + ML 融合实现
└── visualization/
    └── report.py                  # Matplotlib 6子图报告（支持中文字体自动检测）

QuantAlpha_Demo.ipynb              # v1 Jupyter 演示
QuantAlpha_Demo_V2.ipynb           # 🆕 v2 Jupyter 演示（推荐入口，含融合框架）
demo.py                            # Python 脚本版演示
requirements.txt                   # 依赖列表
```

---

## 快速开始

**方式一（推荐）：直接打开 V2 Jupyter Notebook**

```bash
jupyter notebook QuantAlpha_Demo_V2.ipynb
```

包含所有 v2.0 新特性演示（13 个新算子 + 多因子融合完整流程）。

**方式二：打开 V1 Jupyter Notebook（基础功能）**

```bash
jupyter notebook QuantAlpha_Demo.ipynb
```

**方式三：运行 Python 脚本**

```bash
python demo.py
```

---

## 依赖安装

```bash
pip install -r requirements.txt
```

| 依赖包 | 最低版本 | 用途 |
|--------|----------|------|
| numpy | 1.24 | 矩阵运算 |
| pandas | 2.0 | DataFrame 核心操作 |
| matplotlib | 3.7 | 图表绘制 |
| seaborn | 0.12 | 热力图（可选，无则降级） |
| scipy | 1.10 | KDE、正态拟合、OLS、min-variance 优化 |
| **scikit-learn** | **1.3** | **🆕 MLCombiner 机器学习融合（必选）** |

> **可选依赖：** 若需使用 `MLCombiner(model_type='xgboost')`，额外安装：
> ```bash
> pip install xgboost>=1.7.0
> ```
> 未安装时自动降级为 `random_forest`，不影响其他功能。

> **中文字体：** `result.plot()` 会自动检测系统中的 CJK 字体（微软雅黑、SimHei、PingFang SC 等）。若检测不到，中文标签将显示为方块，不影响数值展示。

---

## 模块详解

---

### 1. MockDataGenerator — 模拟数据生成器

**导入路径：** `from quant_alpha_engine import MockDataGenerator`

用于在没有真实数据时快速生成测试用市场数据。基于**几何布朗运动（GBM）**叠加行业共同因子，模拟真实市场的行业相关结构、停牌、涨跌停等特征。

#### 构造参数

```python
MockDataGenerator(
    n_stocks        = 100,          # 股票数量
    n_days          = 504,          # 交易日总数（504 ≈ 2年）
    n_industries    = 10,           # 行业数量
    start_date      = '2022-01-01', # 起始日期（跳过周末）
    seed            = 42,           # 随机种子，None 则每次结果不同
    mu              = 0.08,         # 所有股票年化漂移率的基准值
    sigma           = 0.30,         # 所有股票年化波动率的基准值
    suspended_ratio = 0.04,         # 有停牌历史的股票占比（约4%）
    limit_pct       = 0.099,        # 涨跌停触发阈值（|涨跌幅| >= 9.9%）
)
```

#### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_stocks` | int | 100 | 生成的股票数量。越大，截面覆盖越全，但运算略慢 |
| `n_days` | int | 504 | 交易日数。252 ≈ 1年，504 ≈ 2年，756 ≈ 3年 |
| `n_industries` | int | 10 | 行业数量。每个行业至少分配 2 只股票 |
| `start_date` | str | `'2022-01-01'` | 起始日期。自动跳过周末，不处理节假日 |
| `seed` | int \| None | 42 | 固定随机种子保证复现性；设为 `None` 每次随机 |
| `mu` | float | 0.08 | 年化漂移率基准（每只股票在此基础上 ±4% 随机偏移）|
| `sigma` | float | 0.30 | 年化波动率基准（每只股票在此基础上 ±10% 随机偏移）|
| `suspended_ratio` | float | 0.04 | 历史停牌股票占比，每只停牌股有 1~3 段连续停牌期 |
| `limit_pct` | float | 0.099 | 涨跌停判断阈值，A股通常为 0.099（9.9%）|

#### 方法

```python
data = gen.generate()   # 生成并返回 MockData 对象
```

#### MockData 对象字段

`generate()` 返回一个 `MockData` 数据类，包含以下字段：

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `data.close` | DataFrame | T × N | 收盘价，停牌期间为 NaN |
| `data.open` | DataFrame | T × N | 开盘价 |
| `data.high` | DataFrame | T × N | 最高价 |
| `data.low` | DataFrame | T × N | 最低价 |
| `data.volume` | DataFrame | T × N | 成交量（手，100股/手），与波动正相关 |
| `data.industry` | Series | N | 行业映射，`index=股票代码, values=行业名称` |
| `data.is_suspended` | DataFrame | T × N | 停牌矩阵，`True` 表示当日停牌 |
| `data.is_limit` | DataFrame | T × N | 涨跌停矩阵，`True` 表示触发涨停或跌停 |

所有 DataFrame 的 `index` 为 `DatetimeIndex`（交易日），`columns` 为股票代码（如 `SH600000`）。

#### 使用示例

```python
from quant_alpha_engine import MockDataGenerator

# 基础用法
gen  = MockDataGenerator(n_stocks=100, n_days=504, seed=42)
data = gen.generate()

close    = data.close        # 收盘价矩阵
volume   = data.volume       # 成交量矩阵
industry = data.industry     # 行业映射 Series
is_susp  = data.is_suspended # 停牌矩阵
is_limit = data.is_limit     # 涨跌停矩阵

# 查看数据
print(data.close.shape)                 # (504, 100)
print(data.close.index[:3])             # DatetimeIndex 前3个交易日
print(data.industry.value_counts())     # 各行业股票数量分布

# 熊市模拟（高波动、负漂移）
bear_gen  = MockDataGenerator(mu=-0.05, sigma=0.45, seed=100)
bear_data = bear_gen.generate()
```

---

### 2. AlphaOps — 算子库

**导入路径：** `from quant_alpha_engine.ops import AlphaOps as op`

所有算子均为**静态方法**，输入输出均为 `pd.DataFrame`（Index=时间，Columns=股票），支持任意嵌套组合。

**统一约定：**
- 所有 DataFrame 须按时间升序排列
- 窗口不足时返回 `NaN`（不会报错）
- 截面操作自动忽略 NaN（不参与排名/统计）

---

#### 时序类 (Time-Series)

时序算子在**单只股票的时间序列**上操作，逐列独立计算。

---

##### `Ts_Sum(df, window)` — 滑动窗口求和

```python
result = op.Ts_Sum(df, window)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | DataFrame | 输入矩阵（时间 × 股票） |
| `window` | int | 求和窗口大小（天数），如 `5` 表示过去5天之和 |

**返回：** 与 `df` 同形状的 DataFrame，前 `window-1` 行为 NaN。

```python
# 5日成交量之和（衡量近期活跃度）
vol_sum5 = op.Ts_Sum(data.volume, 5)

# 20日价格涨跌之和（多空力量积累）
pnl_sum = op.Ts_Sum(op.Ts_Delta(data.close, 1), 20)
```

---

##### `Ts_Mean(df, window)` — 滑动窗口均值（移动平均）

```python
result = op.Ts_Mean(df, window)
```

| 参数 | 说明 |
|------|------|
| `window` | 均值窗口大小（天数） |

```python
ma20 = op.Ts_Mean(data.close, 20)   # 20日移动平均
# 价格偏离移动均线的程度
deviation = data.close / op.Ts_Mean(data.close, 20) - 1
```

---

##### `Ts_Max(df, window)` / `Ts_Min(df, window)` — 滑动窗口极值

```python
result = op.Ts_Max(df, window)
result = op.Ts_Min(df, window)
```

```python
# 20日最高价（用于计算价格位置）
high20 = op.Ts_Max(data.high, 20)
low20  = op.Ts_Min(data.low, 20)

# 价格在近20日区间内的位置（0=最低点，1=最高点）
price_pos = (data.close - low20) / (high20 - low20 + 1e-8)
```

---

##### `Ts_Std(df, window)` — 滑动窗口标准差

```python
result = op.Ts_Std(df, window)
```

```python
# 20日价格波动率（用于构造低波动因子）
vol20 = op.Ts_Std(data.close.pct_change(), 20)
# 低波动因子：波动率越低，排名越靠前
low_vol_factor = op.Rank(-vol20)
```

---

##### `Ts_Delta(df, period)` — N 日价差（动量/反转核心）

计算 `df - df.shift(period)`，即当前值与 N 天前的差。

```python
result = op.Ts_Delta(df, period)
```

| 参数 | 说明 |
|------|------|
| `period` | 回看期数（天数） |

```python
# 5日价格变化（短期动量）
delta5  = op.Ts_Delta(data.close, 5)

# 20日价格变化（中期动量）
delta20 = op.Ts_Delta(data.close, 20)

# 反转因子：近期跌幅越大 → 排名越靠前（值越小对应排名越高）
reversal = op.Rank(-op.Ts_Delta(data.close, 5))
```

---

##### `Ts_Delay(df, period)` — 数据滞后 N 天

等价于 `df.shift(period)`。常用于构造"N天前的值"参与计算。

```python
result = op.Ts_Delay(df, period)
```

```python
# 上周同一天的价格
close_5d_ago = op.Ts_Delay(data.close, 5)

# 构造相对强度：当前价格 / 20天前价格
rs = data.close / op.Ts_Delay(data.close, 20)
```

---

##### `Ts_Rank(df, window)` — 时序窗口内百分比排名

当前值在过去 `window` 个观测值中的**分位数排名**（越高表示当前处于历史高位）。

```python
result = op.Ts_Rank(df, window)   # 返回值域 [0, 1]
```

| 参数 | 说明 |
|------|------|
| `window` | 历史回看窗口（天数） |

**与截面 `Rank` 的区别：**
- `Ts_Rank`：单只股票在自身历史中的排名（时间维度）
- `Rank`：所有股票在同一截面日期中的排名（截面维度）

```python
# 当前成交量在近20天中的历史分位（>0.8 表示量能处于历史高位）
vol_rank = op.Ts_Rank(data.volume, 20)

# 价格处于历史高位 + 成交量也处于高位 → 放量创新高
breakout = op.Rank(op.Ts_Rank(data.close, 60) + op.Ts_Rank(data.volume, 20))
```

> ⚠️ `Ts_Rank` 内部使用 `rolling.apply`，对超大矩阵（>500只股票）速度较慢，建议 `window` 不超过 60。

---

##### `Ts_Corr(df1, df2, window)` — 滚动相关系数

计算两个同形状 DataFrame 在滚动窗口内的逐列 **Pearson 相关系数**。

```python
result = op.Ts_Corr(df1, df2, window)   # 返回值域 [-1, 1]
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `df1` | DataFrame | 第一个输入矩阵 |
| `df2` | DataFrame | 第二个输入矩阵，须与 df1 形状相同 |
| `window` | int | 滚动窗口大小 |

> 两个 DataFrame 须有相同的 index 和 columns；若不一致，框架自动取交集对齐。

```python
# 过去10天，成交量与价格的相关系数
# 正值：量价同向（放量上涨）；负值：量价背离（缩量上涨）
vol_price_corr = op.Ts_Corr(data.volume, data.close, 10)

# 量价背离因子：负相关（缩量上涨）排名靠前
factor = op.Rank(-vol_price_corr)

# 行业中性化后使用
factor_neut = op.Neutralize(factor, data.industry)
```

---

#### 截面类 (Cross-Sectional)

截面算子在**同一日期的所有股票**上操作（横向操作），逐行独立计算。

---

##### `Rank(df)` — 截面百分比排名

对每个日期截面，将所有股票的因子值做百分比排名。

```python
result = op.Rank(df)   # 返回值域 [0, 1]
```

- 值越大 → 该股票在当日因子值中排名越靠前
- NaN 值不参与排名，对应位置仍返回 NaN
- 适合消除因子量纲，统一化不同因子的尺度

```python
# 最常用的用法：对原始因子做截面排名，去除极值影响
raw_factor = op.Ts_Delta(data.close, 20)
factor     = op.Rank(raw_factor)        # 值压缩到 [0,1]，均匀分布

# 嵌套：先时序处理，再截面排名
factor = op.Rank(op.Ts_Mean(data.volume, 5) / op.Ts_Mean(data.volume, 20))
```

---

##### `ZScore(df)` — 截面 Z-Score 标准化

对每个日期截面，做标准化：`(value - 截面均值) / 截面标准差`。

```python
result = op.ZScore(df)   # 截面均值≈0，标准差≈1
```

- 适合需要保留因子线性结构（而不是排名信息）的场景
- 对极端值敏感（与 `Rank` 不同，极值会被保留）
- 至少需要 2 个非 NaN 值，否则返回 NaN

```python
# 标准化后的成交量偏离度
vol_zscore = op.ZScore(data.volume)

# 与 Rank 的区别
factor_rank   = op.Rank(raw_factor)    # 均匀分布，对极值不敏感
factor_zscore = op.ZScore(raw_factor)  # 正态分布，保留极值信息
```

---

##### `Scale(df, a=1.0)` — 截面绝对值缩放

将每个截面的因子绝对值之和缩放至 `a`：`sum(|factor_i|) = a`。

```python
result = op.Scale(df, a=1.0)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `a` | 1.0 | 目标绝对值之和 |

**典型用途：** 将因子值直接用作持仓权重（多空组合）时，`Scale(factor, 1)` 确保总权重绝对值为 1，即多头 + 空头总杠杆为 1。

```python
# 将因子直接转为多空权重（正值做多，负值做空，总杠杆=1）
weights_factor = op.Scale(op.ZScore(raw_factor), a=1.0)

# 总绝对权重验证
print(weights_factor.abs().sum(axis=1).mean())  # ≈ 1.0
```

---

#### 特殊类 (Special)

---

##### `Decay_Linear(df, d)` — 线性衰减加权移动平均

WorldQuant 平台的核心算子之一。对过去 `d` 天数据做线性加权平均，**越近的数据权重越大**。

```python
result = op.Decay_Linear(df, d)
```

| 参数 | 说明 |
|------|------|
| `d` | 衰减窗口大小（天数） |

**权重分配：**
- d 天前的数据权重为 **1**（最小）
- d-1 天前权重为 **2**
- ...
- 1 天前权重为 **d-1**
- 当天权重为 **d**（最大）
- 所有权重归一化后求加权均值

**与普通移动平均的区别：**
- `Ts_Mean(df, d)`：等权均值，每天权重 = 1/d
- `Decay_Linear(df, d)`：线性加权，最新数据权重最高，信号衰减更平滑

```python
# 对动量信号做线性衰减平滑（减少信号抖动）
raw_signal = op.Rank(op.Ts_Delta(data.close, 5))
smooth_signal = op.Decay_Linear(raw_signal, d=5)

# 嵌套：先截面排名，再线性衰减，再重新排名
factor = op.Rank(
    op.Decay_Linear(
        op.Rank(op.Ts_Delta(data.close, 10)),
        d=5
    )
)
```

---

##### `Neutralize(df, group_data)` — 行业中性化

通过 OLS 残差法，**剔除因子中的行业共性成分**，保留股票的纯个股特质信号。

```python
result = op.Neutralize(df, group_data)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | DataFrame | 需要中性化的因子矩阵 (T × N) |
| `group_data` | Series 或 DataFrame | 行业映射。`Series(index=股票代码, values=行业名)`（推荐）|

**原理：** 对每个时间截面，以行业哑变量矩阵为自变量做 OLS 回归，取**残差**作为中性化后的因子值。残差中不含行业整体的涨跌信息，只保留个股相对行业的超额表现。

**适用场景：**
- 避免因子大量暴露于某一行业（如科技股、银行股）导致的行业 Beta
- 使因子在行业内部做横向比较，而非跨行业比较

```python
# 量价相关因子，行业中性化
raw_corr = op.Ts_Corr(data.volume, data.close, window=10)
factor_raw  = op.Rank(-raw_corr)

# 中性化：消除行业共性
factor_neut = op.Neutralize(factor_raw, data.industry)

# 中性化后的均值接近 0（行业内多空平衡）
print(factor_neut.mean(axis=1).mean())   # ≈ 0
```

> **注意：** `Neutralize` 内部含逐行循环（每日做一次 OLS），对超长时间序列（>2000天）速度较慢。

---

### 🆕 时序统计算子 (v2)

v2.0 新增 4 个时序统计算子，用于捕捉价格/收益序列的**高阶矩**和**自相关结构**。

---

##### `Ts_Skew(df, window)` — 滚动偏度

```python
result = op.Ts_Skew(df, window)
```

计算过去 `window` 天的三阶标准矩（偏度）。正偏度表示右尾厚（偶有大涨），负偏度表示左尾厚（偶有大跌）。

| 参数 | 说明 |
|------|------|
| `window` | 滚动窗口大小（天数），建议 ≥ 10 |

**返回值域：** (-∞, +∞)，典型值在 [-3, 3] 之间。

```python
daily_ret = data.close.pct_change()
# 滚动偏度（负偏度 → 尾部风险大 → 低配）
factor_skew = op.Rank(-op.Ts_Skew(daily_ret, 20))
```

---

##### `Ts_Kurt(df, window)` — 滚动峰度（超额）

```python
result = op.Ts_Kurt(df, window)
```

计算过去 `window` 天的超额峰度（四阶矩 - 3）。正值表示尖峰厚尾，负值表示扁平薄尾。

```python
# 高峰度意味着极端收益出现频率更高（更危险）
factor_kurt = op.Rank(-op.Ts_Kurt(daily_ret, 20))
```

---

##### `Ts_Autocorr(df, lag, window)` — 滚动自相关系数

```python
result = op.Ts_Autocorr(df, lag, window)   # 返回值域 [-1, 1]
```

计算当前值与 `lag` 天前的值在 `window` 窗口内的滚动 Pearson 相关系数。

| 参数 | 说明 |
|------|------|
| `lag` | 滞后阶数 |
| `window` | 滚动窗口大小 |

```python
# 1阶自相关系数（>0 趋势，<0 反转）
autocorr1 = op.Ts_Autocorr(daily_ret, lag=1, window=20)
# 用自相关做均值回归信号：负自相关 → 反转机会
factor_mean_rev = op.Rank(-autocorr1)
```

---

##### `Ts_Hurst(df, window)` — 赫斯特指数（R/S 分析）

```python
result = op.Ts_Hurst(df, window)   # 返回值域约 [0, 1]
```

通过 R/S 分析（极差/标准差）估计赫斯特指数：
- **H > 0.5**：趋势性（持续性，动量信号有效）
- **H < 0.5**：均值回归（反持续性，反转信号有效）
- **H ≈ 0.5**：随机游走

```python
hurst = op.Ts_Hurst(data.close.pct_change(), window=30)
# 趋势股（H 高）做动量，均值回归股（H 低）做反转
```

> ⚠️ `Ts_Hurst` 使用 `rolling.apply`，对大矩阵较慢，建议 `window` 在 20~60 之间。

---

### 🆕 量价因子算子 (v2)

v2.0 新增 3 个量价融合算子，捕捉**成交量与价格**的微观结构关系。

---

##### `VWAP(close, volume, window)` — 成交量加权均价

```python
result = op.VWAP(close, volume, window)
```

计算过去 `window` 天的成交量加权均价：`Σ(P × V) / ΣV`。

| 参数 | 类型 | 说明 |
|------|------|------|
| `close` | DataFrame | 收盘价矩阵 |
| `volume` | DataFrame | 成交量矩阵 |
| `window` | int | 滚动窗口大小 |

**返回：** 与 `close` 同形状，单位与价格相同。

```python
vwap = op.VWAP(data.close, data.volume, window=10)
# 价格高于 VWAP → 强势；低于 VWAP → 弱势
price_vs_vwap = op.Rank(data.close / vwap - 1)
```

---

##### `PVDeviation(close, volume, window)` — 量价偏离度

```python
result = op.PVDeviation(close, volume, window)
```

计算当前价格相对 VWAP 的偏离程度（标准化）：`(close - VWAP) / rolling_std(close, window)`。

值为正表示价格显著高于 VWAP（做多偏贵），值为负表示价格显著低于 VWAP（潜在低估）。

```python
pv_dev = op.PVDeviation(data.close, data.volume, window=10)
# 负偏离（价格低于 VWAP）→ 均值回归买入信号
factor_pvdev = op.Rank(-pv_dev)
```

---

##### `Amihud(close, volume, window)` — Amihud 非流动性指标

```python
result = op.Amihud(close, volume, window)
```

Amihud（2002）提出的非流动性度量：`mean(|日收益率| / 日成交量)`，值越大表示单位成交量对价格的冲击越大（流动性越差）。

```python
amihud = op.Amihud(data.close, data.volume, window=20)
# 高流动性股票（Amihud 低）通常更受机构偏好
factor_liq = op.Rank(-amihud)
```

---

### 🆕 动量因子算子 (v2)

v2.0 新增 3 个动量因子算子，提供比简单价格动量更精细的信号。

---

##### `RiskAdjMomentum(close, window, vol_window)` — 风险调整动量

```python
result = op.RiskAdjMomentum(close, window=20, vol_window=20)
```

风险调整后的动量因子：`N日累计收益率 / N日滚动波动率`，等价于动量信号的夏普比率。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window` | 20 | 动量计算周期（天数） |
| `vol_window` | 20 | 波动率计算窗口（天数）|

```python
# 高夏普动量：不仅涨了，而且涨得平稳
factor_ram = op.Rank(op.RiskAdjMomentum(data.close, window=20, vol_window=20))
```

---

##### `PricePathQuality(close, window)` — 价格路径质量

```python
result = op.PricePathQuality(close, window)   # 返回值域 [-1, 1]
```

衡量过去 `window` 天价格走势的**单调性与线性度**：
`|Spearman(t, x)| × Pearson(t, x)²`

- 值接近 1：价格单调线性上涨（高质量趋势）
- 值接近 0：价格震荡无方向
- 负值：单调下跌

```python
# 路径质量高的上涨 → 强趋势信号
ppq = op.PricePathQuality(data.close, window=20)
factor_trend = op.Rank(ppq)
```

---

##### `RangeBreakout(close, high, low, window)` — 区间突破位置

```python
result = op.RangeBreakout(close, high, low, window)   # 返回值域 [0, 1]
```

当前收盘价在过去 `window` 天高低区间中的位置：
`(close - rolling_min(low)) / (rolling_max(high) - rolling_min(low))`

- 值接近 1：接近历史高点（突破信号）
- 值接近 0：接近历史低点（超卖信号）

| 参数 | 类型 | 说明 |
|------|------|------|
| `close` | DataFrame | 收盘价 |
| `high` | DataFrame | 最高价 |
| `low` | DataFrame | 最低价 |

```python
breakout = op.RangeBreakout(data.close, data.high, data.low, window=20)
# 价格处于区间高位 → 突破趋势
factor_breakout = op.Rank(breakout)
```

---

### 🆕 技术指标算子 (v2)

v2.0 新增经典技术分析指标作为因子信号，全部向量化实现，支持任意股票矩阵。

---

##### `RSI(close, window=14)` — 相对强弱指数

```python
result = op.RSI(close, window=14)   # 返回值域 [0, 100]
```

Wilder 平滑 RSI：`100 - 100 / (1 + AvgGain / AvgLoss)`，使用 `ewm(span=2×window-1)` 实现。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window` | 14 | RSI 计算周期 |

- **RSI > 70**：超买（可能回调）
- **RSI < 30**：超卖（可能反弹）

```python
rsi = op.RSI(data.close, window=14)
# 超卖反转因子
factor_rsi = op.Rank(-(rsi - 50).abs())  # 距离 50 越近越中性
# 或：超卖做多
factor_oversold = op.Rank(100 - rsi)
```

---

##### `KDJ(close, high, low, n=9, m1=3, m2=3)` — KDJ 指标（K 值）

```python
result = op.KDJ(close, high, low, n=9, m1=3, m2=3)   # 返回 K 值，约 [0, 100]
```

基于 RSV（未成熟随机值）的 EWM 平滑 K 值：
- `RSV = (close - Ln) / (Hn - Ln) × 100`
- `K = EWM(RSV, m1)`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n` | 9 | 区间周期 |
| `m1` | 3 | K 值平滑系数 |
| `m2` | 3 | D 值平滑系数（当前仅返回 K 值）|

```python
k_val = op.KDJ(data.close, data.high, data.low, n=9)
# K 值超买超卖
factor_kdj = op.Rank(50 - k_val)   # 低 K 值排名靠前（超卖）
```

---

##### `MACD(close, fast=12, slow=26, signal=9)` — MACD 柱状图

```python
result = op.MACD(close, fast=12, slow=26, signal=9)   # 返回 MACD 柱状图（Histogram）
```

标准 MACD：
- `DIF = EMA(close, fast) - EMA(close, slow)`
- `DEA = EMA(DIF, signal)`
- **返回柱状图** = `DIF - DEA`（正值上穿为买入信号，负值下穿为卖出信号）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fast` | 12 | 快线 EMA 周期 |
| `slow` | 26 | 慢线 EMA 周期 |
| `signal` | 9 | 信号线 EMA 周期 |

```python
macd_hist = op.MACD(data.close, fast=12, slow=26, signal=9)
# 柱状图由负转正 → 趋势反转信号
factor_macd = op.Rank(macd_hist)
```

---

### 3. VectorEngine — 回测引擎

**导入路径：** `from quant_alpha_engine.backtest import VectorEngine`

矩阵式净值回测引擎，接收因子矩阵和行情数据，输出完整回测结果。

#### 构造参数

```python
VectorEngine(
    factor          = factor_df,     # 必填：因子矩阵
    close           = close_df,      # 必填：收盘价矩阵
    is_suspended    = susp_df,       # 必填：停牌矩阵
    is_limit        = limit_df,      # 必填：涨跌停矩阵
    rebalance_freq  = 1,             # 调仓频率（天数）
    top_n           = 50,            # 持仓股票数
    weight_method   = 'equal',       # 权重方式
    cost_rate       = 0.0015,        # 单边交易成本
    initial_capital = 1_000_000.0,   # 初始资金（仅展示用）
)
```

#### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `factor` | DataFrame | — | **必填**。因子矩阵（T × N），值越大代表该股票越应该被买入 |
| `close` | DataFrame | — | **必填**。收盘价矩阵（T × N），用于计算每日收益率 |
| `is_suspended` | DataFrame | — | **必填**。停牌矩阵（T × N，bool）。停牌股票当日权重自动置零 |
| `is_limit` | DataFrame | — | **必填**。涨跌停矩阵（T × N，bool）。涨跌停股票当日权重自动置零（无法成交）|
| `rebalance_freq` | int | 1 | 调仓频率。`1`=每日调仓，`5`=每周调仓，`21`≈每月调仓。非调仓日持仓不变 |
| `top_n` | int | 50 | 每次调仓时，按因子值排名取前 N 只股票持仓。若可交易股票不足 N 只，则取全部可交易股票 |
| `weight_method` | str | `'equal'` | 持仓权重计算方式，见下方说明 |
| `cost_rate` | float | 0.0015 | 单边交易成本率（手续费 + 滑点）。每次换手的买入或卖出均按此比率扣费 |
| `initial_capital` | float | 1,000,000 | 初始资金，仅用于结果展示，不影响收益率计算 |

**`weight_method` 可选值：**

| 值 | 说明 | 适用场景 |
|----|------|----------|
| `'equal'` | Top-N 股票等权持仓，每只权重 = 1/N | 默认推荐，消除规模偏差 |
| `'factor_weighted'` | Top-N 股票按因子绝对值加权（归一化后），因子值越大权重越高 | 希望因子强信号股票获得更高配置 |

**调仓频率建议：**

| `rebalance_freq` | 等效频率 | 适用因子类型 |
|------------------|----------|-------------|
| 1 | 每日调仓 | 高频动量、量价因子 |
| 5 | 每周调仓 | 中频技术因子（推荐起点）|
| 21 | 每月调仓 | 低频基本面因子 |

#### 方法

```python
result = engine.run()   # 执行回测，返回 BacktestResult
```

`run()` 会自动：
1. 对齐所有输入数据（取共同日期和股票）
2. 计算前向 1 日收益率（严格防未来函数）
3. 在调仓日生成权重，过滤停牌/涨跌停股票
4. 计算每日换手率和交易成本
5. 计算净值序列和全量绩效指标

#### 使用示例

```python
from quant_alpha_engine.backtest import VectorEngine

# 基础用法（每周调仓，等权持仓30只）
engine = VectorEngine(
    factor         = factor,
    close          = data.close,
    is_suspended   = data.is_suspended,
    is_limit       = data.is_limit,
    rebalance_freq = 5,
    top_n          = 30,
    weight_method  = 'equal',
    cost_rate      = 0.0015,
)
result = engine.run()

# 因子加权 + 低换手（每月调仓）
engine2 = VectorEngine(
    factor         = factor,
    close          = data.close,
    is_suspended   = data.is_suspended,
    is_limit       = data.is_limit,
    rebalance_freq = 21,
    top_n          = 20,
    weight_method  = 'factor_weighted',
    cost_rate      = 0.0010,   # 更低成本（机构费率）
)
result2 = engine2.run()
```

---

### 4. BacktestResult — 回测结果对象

`engine.run()` 返回一个 `BacktestResult` 数据类，包含所有中间结果和绩效指标，可直接用于二次分析。

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `result.nav` | Series | 策略净值序列，从 1.0 开始，index=日期 |
| `result.daily_returns` | Series | 扣除交易成本后的日收益率 |
| `result.gross_returns` | Series | 扣除成本**前**的日收益率（毛收益）|
| `result.cost_series` | Series | 每日实际交易成本（换手率 × cost_rate）|
| `result.weights` | DataFrame | 持仓权重矩阵（T × N），已含前向填充 |
| `result.turnover` | Series | 每日单边换手率 = `sum(|w_t - w_{t-1}|) / 2` |
| `result.ic_series` | Series | 每日截面 Rank IC 值 |
| `result.forward_returns` | DataFrame | 前向1日收益率矩阵（T × N）|
| `result.factor` | DataFrame | 对齐后的因子矩阵（T × N）|
| `result.metrics` | dict | 所有绩效指标字典（见下方指标说明）|
| `result.rebalance_dates` | list | 实际调仓日期列表 |

#### 方法

```python
result.print_summary()              # 控制台打印 Unicode 格式指标表格
result.plot()                       # 弹窗显示 6 子图分析报告
result.plot(save_path='out.png')    # 保存报告为 PNG 文件（dpi=150）
```

#### 二次分析示例

```python
# 查看净值序列
result.nav.plot(title='策略净值')

# 查看权重分布
result.weights.sum(axis=1)   # 每日总权重（应接近1）

# 找到持仓最多的股票
result.weights.mean().nlargest(10)   # 历史平均权重最高的10只股票

# 提取 IC 序列做统计
ic = result.ic_series.dropna()
print(f"IC 均值: {ic.mean():.4f}, ICIR: {ic.mean()/ic.std()*252**0.5:.3f}")

# 访问原始指标字典
print(result.metrics['Sharpe_Ratio'])
print(result.metrics['IC_Mean'])

# 计算累计成本
cumcost = result.cost_series.cumsum()
```

---

### 5. Performance — 绩效指标计算

**导入路径：** `from quant_alpha_engine.backtest.performance import Performance`

独立的绩效计算工具类，所有方法为静态方法，可对任意序列调用，不依赖回测引擎。

#### 方法列表

---

##### `Performance.calc_annualized_return(nav)` — 年化收益率

```python
ann_ret = Performance.calc_annualized_return(nav)
# 公式：(nav_末 / nav_初)^(252/T) - 1
```

| 参数 | 说明 |
|------|------|
| `nav` | 净值序列（pd.Series，从1.0开始）|

**返回：** float，如 `0.15` 表示年化 15%

---

##### `Performance.calc_annualized_volatility(returns)` — 年化波动率

```python
ann_vol = Performance.calc_annualized_volatility(returns)
# 公式：std(r) * sqrt(252)
```

**返回：** float，如 `0.20` 表示年化 20%

---

##### `Performance.calc_sharpe(returns, risk_free=0.0, annualize=True)` — 夏普比率

```python
sharpe = Performance.calc_sharpe(returns, risk_free=0.03)
# 公式：(mean(r) - rf/252) / std(r) * sqrt(252)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `returns` | — | 日收益率序列 |
| `risk_free` | 0.0 | 无风险利率（**年化**），如 0.03 表示 3% |
| `annualize` | True | 是否年化 |

**返回：** float，越大越好（通常 >1 为优秀，>2 为卓越）

---

##### `Performance.calc_max_drawdown(nav)` — 最大回撤

```python
mdd = Performance.calc_max_drawdown(nav)
# 公式：min((nav_t - max(nav_{0..t})) / max(nav_{0..t}))
```

**返回：** float（**负数**），如 `-0.15` 表示最大回撤 -15%

---

##### `Performance.calc_calmar(nav)` — Calmar 比率

```python
calmar = Performance.calc_calmar(nav)
# 公式：年化收益率 / |最大回撤|
```

**返回：** float，衡量单位回撤风险获得的年化收益，越大越好

---

##### `Performance.calc_ic_series(factor, forward_returns)` — 逐日 IC 序列

```python
ic_series = Performance.calc_ic_series(factor_df, fwd_ret_df)
```

| 参数 | 说明 |
|------|------|
| `factor` | 因子矩阵（T × N） |
| `forward_returns` | 对应的前向收益矩阵（T × N），须与 factor 对齐 |

**IC 计算方式：** 每日截面 Rank IC = Spearman 相关系数（向量化实现，等价于对排名后的因子和收益做 Pearson 相关）

**返回：** pd.Series，每日的 IC 值，index=日期

---

##### `Performance.calc_ic_stats(ic_series)` — IC 统计摘要

```python
stats = Performance.calc_ic_stats(ic_series)
```

**返回字典：**

| Key | 说明 |
|-----|------|
| `IC_Mean` | IC 均值，通常优秀因子的 |IC| > 0.03 |
| `IC_Std` | IC 标准差，反映 IC 稳定性 |
| `ICIR` | IC 信息比率 = IC_Mean / IC_Std × √252，通常 >0.5 为优秀 |
| `IC_Positive_Ratio` | IC > 0 的比例（胜率），通常 >55% 为优秀 |
| `IC_t_stat` | IC 均值的 t 统计量，|t| > 2 表示统计显著 |

---

##### `Performance.calc_turnover(weights)` — 换手率序列

```python
turnover = Performance.calc_turnover(weights_df)
# 公式：sum(|w_t - w_{t-1}|) / 2
```

**返回：** pd.Series，每日单边换手率（0~1之间，如 0.05 表示 5%）

---

##### `Performance.calc_fitness(sharpe, nav, turnover)` — Fitness 指标

```python
fitness = Performance.calc_fitness(sharpe, nav, turnover)
# 公式：Sharpe × sqrt(|年化收益率| / 日均换手率)
```

WorldQuant 用于综合衡量因子**收益质量 × 稳定性 / 换手成本**的综合指标。

**返回：** float，通常 >1.0 为有价值的因子

---

##### `Performance.summary(...)` — 完整指标汇总

```python
metrics = Performance.summary(
    nav             = nav,
    daily_returns   = returns,
    weights         = weights,
    factor          = factor,
    forward_returns = fwd_ret,
    cost_series     = cost_series,   # 可选
)
```

**返回字典（所有键名）：**

```python
{
    '年化收益率': 0.1234,    # 如 0.1234 表示 12.34%
    '年化波动率': 0.0856,
    'Sharpe_Ratio': 1.44,
    'Calmar_Ratio': 0.88,
    '最大回撤': -0.1567,     # 负数
    'IC_Mean': 0.053,
    'IC_Std': 0.089,
    'ICIR': 1.23,
    'IC_胜率': 0.583,        # 如 0.583 表示 58.3%
    'IC_t统计量': 3.45,
    '日均换手率': 0.056,
    '年化手续费': 0.0123,
    'Fitness': 0.89,
}
```

#### 独立使用示例

```python
from quant_alpha_engine.backtest.performance import Performance
import pandas as pd

# 对自己的收益序列计算指标
my_returns = pd.Series([0.001, -0.002, 0.003, ...], index=dates)
my_nav     = (1 + my_returns).cumprod()

sharpe = Performance.calc_sharpe(my_returns, risk_free=0.03)
mdd    = Performance.calc_max_drawdown(my_nav)
print(f"Sharpe: {sharpe:.3f}, MaxDD: {mdd*100:.1f}%")

# 计算 IC
ic = Performance.calc_ic_series(my_factor, my_fwd_returns)
stats = Performance.calc_ic_stats(ic)
print(f"IC 均值: {stats['IC_Mean']:.4f}, ICIR: {stats['ICIR']:.3f}")
```

---

### 6. Report — 可视化报告

**导入路径：** `from quant_alpha_engine.visualization.report import Report`

通常通过 `result.plot()` 间接调用，也可直接调用 `Report.plot(result)`。

#### `Report.plot(result, save_path=None, benchmark_seed=2024)`

```python
Report.plot(result)                            # 弹窗展示
Report.plot(result, save_path='report.png')    # 保存为 PNG（dpi=150）
Report.plot(result, benchmark_seed=42)         # 指定基准随机种子
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `result` | — | BacktestResult 对象 |
| `save_path` | None | 保存路径（如 `'./output/factor1.png'`）。为 None 时弹窗展示 |
| `benchmark_seed` | 2024 | 模拟基准（随机游走）的随机种子，固定后每次基准相同 |

#### 6 子图布局

```
┌─────────────────┬──────────────────┬──────────────────┐
│  📈 净值曲线     │  🗓️ 月度收益热力图 │   📊 IC 时序      │
│  策略 vs 基准    │   行=年，列=月    │  正绿负红 + 均值线 │
├─────────────────┼──────────────────┼──────────────────┤
│  📉 日收益率分布 │   🔄 换手率序列   │   📐 IC 分布      │
│  直方图+KDE+正态 │  折线+均值+调仓日 │ 直方图+正态+胜率  │
└─────────────────┴──────────────────┴──────────────────┘
                  顶部指标摘要条（8项核心指标）
```

| 子图 | 内容 | 关键信息 |
|------|------|----------|
| 净值曲线 | 策略净值（红）+ 模拟基准（灰虚线）+ 回撤阴影 | 最大回撤标注箭头 |
| 月度收益热力图 | 每月收益率，红绿配色（绿=正收益，红=负收益）| 直观看季节性规律 |
| IC 时序 | 每日 IC 柱状图（正绿负红）+ IC 均值虚线 | 左上角标注 ICIR |
| 日收益率分布 | 实际分布直方图 + KDE + 正态分布对比曲线 | 尖峰厚尾特征 |
| 换手率序列 | 非零换手率折线 + 均值虚线 + 调仓日标记（灰色竖线）| 了解成本节奏 |
| IC 分布 | IC 值分布直方图 + 正态拟合 + 胜率标注 | IC 分布偏正性 |

---

### 7. Fusion — 多因子融合框架 (v2)

**导入路径：**
```python
from quant_alpha_engine.fusion import Labeler, StatisticalCombiner, MLCombiner
# 或通过根包直接导入：
from quant_alpha_engine import Labeler, StatisticalCombiner, MLCombiner
```

多因子融合框架提供两种范式将多个单因子合成为一个综合因子，并直接输出标准 `BacktestResult`，无缝对接现有评估体系。

**完整融合流程：**

```python
from quant_alpha_engine import MockDataGenerator, Labeler, StatisticalCombiner, MLCombiner
from quant_alpha_engine.ops import AlphaOps as op

# 1. 准备数据
data   = MockDataGenerator(n_stocks=100, n_days=504).generate()
close  = data.close

# 2. 构造多个单因子
f1 = op.Rank(op.Ts_Delta(close, 5))
f2 = op.Rank(-op.Ts_Corr(data.volume, close, 10))
f3 = op.Rank(op.MACD(close))

# 3. 生成标签
label = Labeler().set_label(target='close', horizon=5, data={'close': close})

# 4. 统计融合（IC 加权）
stat = StatisticalCombiner('ic_weighted').fit([f1, f2, f3], label)
result = stat.evaluate(
    [f1, f2, f3],
    close=close, is_suspended=data.is_suspended, is_limit=data.is_limit,
    rebalance_freq=5, top_n=30,
)
result.print_summary()
result.plot()

# 5. ML 融合（Ridge，防未来函数）
ml = MLCombiner('ridge', min_train_periods=60, refit_freq=20)
ml.fit([f1, f2, f3], label)
result2 = ml.evaluate(
    [f1, f2, f3],
    close=close, is_suspended=data.is_suspended, is_limit=data.is_limit,
    rebalance_freq=5, top_n=30,
)
result2.plot()
print(ml.feature_importances_)  # 查看各因子重要性
```

---

#### 7.1 Labeler — 标签生成器

**导入路径：** `from quant_alpha_engine.fusion import Labeler`

生成用于多因子融合的**前向收益率标签**（监督学习的目标 Y）。

##### `set_label(target, horizon, method, data, custom_label)` — 生成标签

```python
label = Labeler().set_label(
    target       = 'close',    # 目标价格字段：'close' | 'open' | 'vwap' 等
    horizon      = 5,          # 前向天数（预测几天后的收益）
    method       = 'return',   # 收益计算方式：'return' | 'log_return'
    data         = {'close': close_df},  # 含目标价格矩阵的字典
    custom_label = None,       # 用户自定义 Y (T×N DataFrame)，优先级最高
)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `target` | `'close'` | 价格字段名，需与 `data` 字典中的 key 对应 |
| `horizon` | `1` | 前向天数，如 `5` 表示预测 5 日后的收益率 |
| `method` | `'return'` | `'return'`：简单收益率 `P_{t+h}/P_t - 1`；`'log_return'`：对数收益率 |
| `data` | `None` | 价格数据字典，如 `{'close': close_df}` |
| `custom_label` | `None` | 若提供，直接返回该 DataFrame（跳过内置计算）|

**返回：** pd.DataFrame（T × N），与输入价格矩阵形状相同，前 `horizon` 行为 NaN。

```python
# 方式一：内置标签（5日简单收益率）
y = Labeler().set_label(target='close', horizon=5, data={'close': close})

# 方式二：对数收益率标签
y_log = Labeler().set_label(target='close', horizon=5,
                             method='log_return', data={'close': close})

# 方式三：用户自定义标签
my_y = (close.shift(-3) / close - 1)   # 3日收益率
y_custom = Labeler().set_label(custom_label=my_y)

# 静态工厂方法（直接从价格计算）
y_static = Labeler.from_price(close, horizon=5, method='return')
```

---

#### 7.2 StatisticalCombiner — 统计融合

**导入路径：** `from quant_alpha_engine.fusion import StatisticalCombiner`

基于统计权重的多因子合成，支持三种权重计算方式。

##### 构造参数

```python
StatisticalCombiner(method='equal')
```

| `method` | 说明 | 适用场景 |
|----------|------|----------|
| `'equal'` | 等权合成 | 快速基准，无需训练标签 |
| `'ic_weighted'` | 按 IC 均值加权 | 预测力强的因子获得更高权重 |
| `'min_variance'` | 最小方差优化（SLSQP） | 追求合成因子的稳定性 |

> 所有方法在合成前自动对每个因子做 `rank(axis=1, pct=True)` 处理（消除量纲）。

##### 方法

```python
combiner = StatisticalCombiner('ic_weighted')

# 训练（计算权重）
combiner.fit(factors=[f1, f2, f3], label=y)

# 预测（生成合成因子 T×N）
composite = combiner.predict([f1, f2, f3])

# 一键回测评估
result = combiner.evaluate(
    factors      = [f1, f2, f3],
    close        = close,
    is_suspended = data.is_suspended,
    is_limit     = data.is_limit,
    rebalance_freq = 5,
    top_n        = 30,
    weight_method  = 'equal',
    cost_rate      = 0.0015,
)

# 查看权重
print(combiner.weights_)          # np.ndarray，各因子权重
print(combiner.ic_matrix_)        # DataFrame，每日各因子 IC 值

# 持久化
combiner.save('stat_combiner.pkl')
loaded = StatisticalCombiner.load('stat_combiner.pkl')
```

##### 使用示例

```python
from quant_alpha_engine.fusion import StatisticalCombiner

# 等权合成（无需标签）
equal = StatisticalCombiner('equal').fit([f1, f2, f3], y)
print("等权权重:", equal.weights_)      # [0.333, 0.333, 0.333]

# IC 加权（预测力强的因子权重更高）
ic_w = StatisticalCombiner('ic_weighted').fit([f1, f2, f3], y)
print("IC加权权重:", ic_w.weights_)    # 例：[0.45, 0.35, 0.20]

# 最小方差合成（减少合成因子波动）
min_var = StatisticalCombiner('min_variance').fit([f1, f2, f3], y)
result = min_var.evaluate([f1, f2, f3], close, data.is_suspended, data.is_limit,
                          rebalance_freq=5, top_n=30)
result.print_summary()
```

---

#### 7.3 MLCombiner — 机器学习融合

**导入路径：** `from quant_alpha_engine.fusion import MLCombiner`

使用机器学习模型学习因子与未来收益的关系，通过 **Expanding Window** 方式严格防止未来函数。

##### 构造参数

```python
MLCombiner(
    model_type         = 'ridge',   # 模型类型
    min_train_periods  = 60,        # 最小训练期（前 N 天为 NaN，不预测）
    refit_freq         = 20,        # 每隔 N 天重新训练一次
    ridge_alpha        = 1.0,       # Ridge 正则化强度
    rf_n_estimators    = 100,       # RandomForest 树的数量
    rf_max_depth       = 5,         # RandomForest 最大深度
    xgb_n_estimators   = 100,       # XGBoost 树的数量（需安装 xgboost）
    xgb_max_depth      = 3,         # XGBoost 最大深度
    xgb_learning_rate  = 0.1,       # XGBoost 学习率
)
```

| `model_type` | 说明 | 特征重要性来源 |
|-------------|------|--------------|
| `'linear'` | 普通线性回归 | `\|coef_\|` 归一化 |
| `'ridge'` | Ridge 回归（L2 正则化）| `\|coef_\|` 归一化 |
| `'random_forest'` | 随机森林 | `feature_importances_` |
| `'xgboost'` | XGBoost（需额外安装）| `feature_importances_`，未安装时自动降级为 `random_forest` |

##### Expanding Window 防未来函数原则

```
训练期（前 min_train_periods 天）→ 输出全为 NaN

Day  0 ~ 59  → 积累期，无预测
Day 60 ~ 79  → 用 [0, 60) 训练 → 预测 [60, 80)
Day 80 ~ 99  → 用 [0, 80) 训练 → 预测 [80, 100)
Day 100~119  → 用 [0, 100) 训练 → 预测 [100, 120)
...（训练集持续扩大，永不使用未来数据）
```

##### 方法

```python
ml = MLCombiner('ridge', min_train_periods=60, refit_freq=20)

# 训练
ml.fit(factors=[f1, f2, f3], label=y)

# 预测（生成合成因子 T×N，前 min_train_periods 行为 NaN）
pred = ml.predict([f1, f2, f3])
assert pred.iloc[:60].isna().all().all()   # 严格验证无未来函数

# 查看各因子重要性
print(ml.feature_importances_)   # pd.Series，index = ['f0', 'f1', 'f2']

# 一键回测评估
result = ml.evaluate(
    [f1, f2, f3],
    close=close, is_suspended=data.is_suspended, is_limit=data.is_limit,
    rebalance_freq=5, top_n=30,
)
result.plot()

# 持久化（两种模式）
ml.save('ml_combiner.pkl')                            # 含预测缓存（文件较大）
ml.save('ml_combiner_lite.pkl', save_predictions=False)  # 仅模型权重（文件更小）
loaded = MLCombiner.load('ml_combiner.pkl')
```

##### 使用示例

```python
from quant_alpha_engine.fusion import MLCombiner

# Ridge 融合
ml_ridge = MLCombiner('ridge', min_train_periods=60, refit_freq=20, ridge_alpha=1.0)
ml_ridge.fit([f1, f2, f3], y)

# 验证无未来函数
pred = ml_ridge.predict([f1, f2, f3])
assert pred.iloc[:60].isna().all().all()
print(f"有效预测天数: {pred.dropna(how='all').shape[0]}")

# 查看特征重要性
print(ml_ridge.feature_importances_)

# RandomForest 融合（捕捉非线性关系）
ml_rf = MLCombiner('random_forest', min_train_periods=120, refit_freq=60,
                   rf_n_estimators=100, rf_max_depth=5)
ml_rf.fit([f1, f2, f3], y)
result_rf = ml_rf.evaluate(
    [f1, f2, f3],
    close=close, is_suspended=data.is_suspended, is_limit=data.is_limit,
    rebalance_freq=5, top_n=30,
)
result_rf.print_summary()
```

---

## 指标说明

| 指标 | 计算公式 | 参考范围 | 含义 |
|------|----------|----------|------|
| 年化收益率 | `(nav_末/nav_初)^(252/T) - 1` | >5% | 策略年化超额收益 |
| 年化波动率 | `std(日收益) × √252` | <20% | 收益的不稳定程度 |
| **Sharpe Ratio** | `年化超额收益 / 年化波动率` | **>1.0** | 风险调整后收益，越高越好 |
| Calmar Ratio | `年化收益率 / \|最大回撤\|` | >0.5 | 单位回撤风险的年化收益 |
| **最大回撤** | `min((nav_t - nav_高点) / nav_高点)` | **>-30%** | 历史最大亏损幅度，越小越好 |
| **IC 均值** | 每日 Rank IC 的均值 | **>0.03** | 因子预测力，越高越好 |
| IC 标准差 | 每日 IC 的标准差 | — | IC 稳定性，越低越好 |
| **ICIR** | `IC均值 / IC标准差 × √252` | **>0.5** | IC 信息比率，综合衡量稳定预测力 |
| IC 胜率 | IC > 0 的天数占比 | >55% | 因子方向正确的概率 |
| IC t-stat | IC 均值的 t 统计量 | `\|t\|>2` | IC 统计显著性，>2 表示显著 |
| 日均换手率 | `mean(Σ\|w_t - w_{t-1}\| / 2)` | <10% | 每日平均换手比例，越低成本越小 |
| 年化手续费 | `日均交易成本 × 252` | — | 每年因交易损耗的收益 |
| **Fitness** | `Sharpe × √(\|年化收益\| / 日均换手)` | **>1.0** | WorldQuant 综合评估指标 |

---

## 使用真实数据

将框架应用到真实数据时，只需将数据组织为正确格式的 DataFrame：

### 数据格式要求

```python
# 所有 DataFrame 须满足：
# - index: pd.DatetimeIndex，时间升序
# - columns: 股票代码（字符串），命名一致

import pandas as pd

# 价格数据（T × N）
close = pd.DataFrame(...)    # index=日期, columns=股票代码

# 停牌矩阵（T × N），bool 类型
is_suspended = pd.DataFrame(...).astype(bool)

# 涨跌停矩阵（T × N），bool 类型
# 可以合并涨停和跌停：is_limit = is_limit_up | is_limit_down
is_limit = (is_limit_up | is_limit_down).astype(bool)

# 行业映射（静态）
# index=股票代码, values=行业名称字符串
industry = pd.Series({
    '000001.SZ': '银行',
    '000002.SZ': '房地产',
    ...
})
```

### 完整替换模板

```python
import pandas as pd
from quant_alpha_engine.ops import AlphaOps as op
from quant_alpha_engine.backtest import VectorEngine

# ── 加载您的数据 ──────────────────────────────────
close        = pd.read_csv('close.csv',        index_col=0, parse_dates=True)
volume       = pd.read_csv('volume.csv',       index_col=0, parse_dates=True)
is_suspended = pd.read_csv('suspended.csv',    index_col=0, parse_dates=True).astype(bool)
is_limit_up  = pd.read_csv('limit_up.csv',     index_col=0, parse_dates=True).astype(bool)
is_limit_dn  = pd.read_csv('limit_down.csv',   index_col=0, parse_dates=True).astype(bool)
is_limit     = is_limit_up | is_limit_dn

industry     = pd.read_csv('industry.csv', index_col=0).squeeze()  # Series

# ── 构造您的因子 ──────────────────────────────────
factor = op.Neutralize(
    op.Rank(-op.Ts_Corr(volume, close, window=10)),
    industry
)

# ── 一行回测 ──────────────────────────────────────
result = VectorEngine(
    factor         = factor,
    close          = close,
    is_suspended   = is_suspended,
    is_limit       = is_limit,
    rebalance_freq = 5,
    top_n          = 50,
    weight_method  = 'equal',
    cost_rate      = 0.0015,
).run()

result.print_summary()
result.plot(save_path='my_factor_report.png')
```

---

## 因子构造示例集

以下示例展示常见因子类型的算子组合方式，可直接复用或参考修改：

```python
from quant_alpha_engine.ops import AlphaOps as op

# ── 反转类 ─────────────────────────────────────────────────────────────

# 短期反转（5日）
factor_rev5 = op.Rank(-op.Ts_Delta(close, 5))

# 中期反转（20日）+ 行业中性化
factor_rev20 = op.Neutralize(op.Rank(-op.Ts_Delta(close, 20)), industry)

# ── 动量类 ─────────────────────────────────────────────────────────────

# 月度动量（跳过最近5天，避免短期反转）
momentum = close / op.Ts_Delay(close, 21) - close / op.Ts_Delay(close, 5)
factor_mom = op.Rank(momentum)

# 线性衰减平滑的动量
factor_decay_mom = op.Rank(op.Decay_Linear(op.Rank(op.Ts_Delta(close, 10)), d=5))

# 🆕 风险调整动量（v2）
factor_ram = op.Rank(op.RiskAdjMomentum(close, window=20, vol_window=20))

# 🆕 价格路径质量（v2）— 单调线性趋势
factor_ppq = op.Rank(op.PricePathQuality(close, window=20))

# ── 量价类 ─────────────────────────────────────────────────────────────

# 量价背离（缩量上涨为正信号）
factor_vp = op.Neutralize(op.Rank(-op.Ts_Corr(volume, close, 10)), industry)

# 成交量时序分位（近期成交量处于历史低位）
factor_vol_rank = op.Rank(-op.Ts_Rank(volume, 20))

# 放量创新高（价格创新高 + 成交量也创高）
price_rank  = op.Ts_Rank(close, 60)
vol_rank    = op.Ts_Rank(volume, 20)
factor_breakout = op.Rank(price_rank + vol_rank)

# 🆕 VWAP 偏离度（v2）
factor_pvdev = op.Rank(-op.PVDeviation(close, volume, window=10))

# 🆕 Amihud 流动性（v2）— 高流动性股票
factor_liq = op.Rank(-op.Amihud(close, volume, window=20))

# ── 波动率类 ────────────────────────────────────────────────────────────

# 低波动因子（波动率越小越好）
daily_ret = close.pct_change()
factor_lowvol = op.Rank(-op.Ts_Std(daily_ret, 20))

# 🆕 高阶矩因子（v2）— 低负偏度（尾部风险小）
factor_skew = op.Rank(-op.Ts_Skew(daily_ret, 20))

# ── 技术指标类 ──────────────────────────────────────────────────────────

# 🆕 RSI 超卖因子（v2）
factor_rsi = op.Rank(100 - op.RSI(close, window=14))   # RSI 越低 → 越超卖

# 🆕 MACD 柱状图（v2）
factor_macd = op.Rank(op.MACD(close, fast=12, slow=26, signal=9))

# 🆕 KDJ 超卖（v2）
factor_kdj = op.Rank(50 - op.KDJ(close, high, low, n=9))

# 🆕 区间突破（v2）
factor_rng = op.Rank(op.RangeBreakout(close, high, low, window=20))

# ── 价格位置类 ──────────────────────────────────────────────────────────

# Williams %R（价格在近20日区间内的位置，越低越超卖）
high20 = op.Ts_Max(high, 20)
low20  = op.Ts_Min(low,  20)
wr     = (close - high20) / (high20 - low20 + 1e-8)
factor_wr = op.Rank(-wr)   # 超卖排名靠前

# ── 多因子合成（手动等权） ────────────────────────────────────────────────

# 等权合成（先分别 Rank 消除量纲，再加权）
alpha_combo = (
    0.4 * op.Rank(factor_rev5) +
    0.3 * op.Rank(factor_vp) +
    0.3 * op.Rank(factor_lowvol)
)

# ── 🆕 多因子融合（fusion 模块）─────────────────────────────────────────

from quant_alpha_engine.fusion import Labeler, StatisticalCombiner, MLCombiner

label = Labeler().set_label(target='close', horizon=5, data={'close': close})

# IC 加权统计融合
stat = StatisticalCombiner('ic_weighted').fit([factor_rev5, factor_vp, factor_rsi], label)
result = stat.evaluate([factor_rev5, factor_vp, factor_rsi],
                       close, data.is_suspended, data.is_limit,
                       rebalance_freq=5, top_n=30)
result.print_summary()

# Ridge ML 融合
ml = MLCombiner('ridge', min_train_periods=60, refit_freq=20)
ml.fit([factor_rev5, factor_vp, factor_rsi], label)
result2 = ml.evaluate([factor_rev5, factor_vp, factor_rsi],
                      close, data.is_suspended, data.is_limit,
                      rebalance_freq=5, top_n=30)
result2.plot()
```

---

## 常见问题

**Q: 回测时提示 `数据规模：N 个交易日 × M 只股票`，M 比预期少？**

A: `VectorEngine` 会自动取 `factor`、`close`、`is_suspended`、`is_limit` 四个输入的**共同股票列交集**。请检查各 DataFrame 的 columns 是否完全一致（大小写、格式需相同）。

---

**Q: `Ts_Rank` 运行很慢？**

A: `Ts_Rank` 内部使用 `rolling.apply`（Python 层循环），对大矩阵性能有限。优化建议：
1. 减少股票数量或缩短时间序列
2. 将 `window` 控制在 60 以内
3. 考虑用 `Rank(df)` 替代（截面排名速度更快）

---

**Q: 想对因子做 IC 分析，但不想走完整回测流程？**

A: 直接使用 `Performance` 模块：

```python
from quant_alpha_engine.backtest.performance import Performance

# 计算前向1日收益率
fwd_ret = close.shift(-1) / close - 1

# 计算 IC 序列
ic = Performance.calc_ic_series(my_factor, fwd_ret)
stats = Performance.calc_ic_stats(ic)
print(stats)
```

---

**Q: 如何保存回测报告图片？**

```python
result.plot(save_path='./reports/factor1_report.png')
# 或
from quant_alpha_engine.visualization.report import Report
Report.plot(result, save_path='./reports/factor1_report.png')
```

---

**Q: `Neutralize` 中性化后因子均值不为零？**

A: 正常现象。`Neutralize` 使用 OLS 残差，残差在每个行业内均值为零，但**跨行业的整体均值**不一定为零。若需整体均值为零，可在中性化后再做一次 `ZScore`：

```python
factor_neut = op.Neutralize(raw_factor, industry)
factor_final = op.ZScore(factor_neut)   # 确保截面均值=0
```

---

**Q: 如何模拟不同市场环境（牛市/熊市/震荡）？**

```python
# 牛市（高漂移 + 低波动）
bull = MockDataGenerator(mu=0.20, sigma=0.18, seed=1).generate()

# 熊市（负漂移 + 高波动）
bear = MockDataGenerator(mu=-0.10, sigma=0.45, seed=2).generate()

# 震荡市（零漂移 + 中等波动）
flat = MockDataGenerator(mu=0.00, sigma=0.25, seed=3).generate()
```

---

**Q: 如何在 Jupyter Notebook 中保存图表而不弹窗？**

```python
import matplotlib
matplotlib.use('Agg')   # 在 import pyplot 之前设置非交互后端

result.plot(save_path='report.png')
```

或在 Notebook 顶部使用 `%matplotlib inline` 时，直接用 `save_path` 参数保存。

---

**Q: 🆕 `MLCombiner` 预测前几行全是 NaN，这是正常的吗？**

A: 完全正常，这是**防未来函数**设计的核心保证。前 `min_train_periods` 行（默认60天）没有足够的历史数据来训练模型，因此输出 NaN。`VectorEngine` 会自动通过 `ffill` 填充，回测期间权重从第一个有效预测日开始建立。

```python
ml = MLCombiner('ridge', min_train_periods=60)
ml.fit([f1, f2, f3], y)
pred = ml.predict([f1, f2, f3])
# 前 60 行应全为 NaN（严格验证）
assert pred.iloc[:60].isna().all().all()
# 第 60 行之后有有效预测
print(f"积累期：前 {pred.isna().all(axis=1).sum()} 行")
```

---

**Q: 🆕 `StatisticalCombiner` 和 `MLCombiner` 的 `evaluate()` 需要哪些参数？**

A: 与 `VectorEngine` 完全一致（除了 `factor` 参数被 `factors` 列表替代）：

```python
result = combiner.evaluate(
    factors        = [f1, f2, f3],    # 因子列表
    close          = close_df,         # 必填
    is_suspended   = susp_df,          # 必填
    is_limit       = limit_df,         # 必填
    rebalance_freq = 5,                # 可选，默认 1
    top_n          = 30,               # 可选，默认 50
    weight_method  = 'equal',          # 可选，默认 'equal'
    cost_rate      = 0.0015,           # 可选，默认 0.0015
)
```

---

**Q: 🆕 如何保存和加载训练好的融合模型？**

```python
# 保存
stat.save('my_combiner.pkl')
ml.save('my_ml_combiner.pkl')
ml.save('my_ml_light.pkl', save_predictions=False)  # 不含预测缓存，文件更小

# 加载
from quant_alpha_engine.fusion import StatisticalCombiner, MLCombiner
stat2 = StatisticalCombiner.load('my_combiner.pkl')
ml2   = MLCombiner.load('my_ml_combiner.pkl')

# 验证加载成功
assert stat2._is_fitted
assert ml2._is_fitted
```

---

**Q: 🆕 新算子（RSI、MACD、KDJ 等）的返回值是什么类型？**

A: 所有新算子均返回 `pd.DataFrame`，形状与输入相同（T × N，Index=日期，Columns=股票代码），可直接作为因子传入 `VectorEngine` 或融合框架。不需要逐股票循环调用。

```python
from quant_alpha_engine.ops import AlphaOps as op

# 批量计算所有股票的 RSI（T × N）
rsi = op.RSI(close, window=14)          # DataFrame
macd = op.MACD(close)                   # DataFrame
hurst = op.Ts_Hurst(close.pct_change(), window=30)  # DataFrame

# 直接用于回测
factor = op.Rank(rsi)
result = VectorEngine(factor=factor, close=close, ...).run()
```
