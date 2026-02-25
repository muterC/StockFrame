# AlphaOps 因子算子库 — 完整参考文档

> 本文档包含 QuantAlpha Engine 所有预置算子的详细说明、参数定义和使用示例。
> 主框架文档请参阅 [README.md](./README.md)。

**导入路径：** `from quant_alpha_engine.ops import AlphaOps as op`

所有算子均为**静态方法**，输入输出均为 `pd.DataFrame`（Index=时间，Columns=股票），支持任意嵌套组合。

**统一约定：**
- 所有 DataFrame 须按时间升序排列
- 窗口不足时返回 `NaN`（不会报错）
- 截面操作自动忽略 NaN（不参与排名/统计）

---

## 目录

- [时序类算子 (v1)](#时序类-time-series-v1)
- [截面类算子 (v1)](#截面类-cross-sectional-v1)
- [特殊类算子 (v1)](#特殊类-special-v1)
- [🆕 时序统计算子 (v2)](#-时序统计算子-v2)
- [🆕 量价因子算子 (v2)](#-量价因子算子-v2)
- [🆕 动量因子算子 (v2)](#-动量因子算子-v2)
- [🆕 技术指标算子 (v2)](#-技术指标算子-v2)
- [🆕 缩量稳价因子 (V3)](#-缩量稳价因子-v3)

---

## 时序类 (Time-Series) (v1)

对每只股票独立做时间序列滚动计算。

---

##### `Ts_Sum(df, window)` — 时序滚动求和

```python
result = op.Ts_Sum(df, window)
```

对每只股票计算过去 `window` 天的**滚动求和**。

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | DataFrame | 输入矩阵（T × N） |
| `window` | int | 滚动窗口大小（天数） |

**返回值域：** 与输入同量纲，前 `window-1` 行为 NaN。

```python
# 过去10日成交量之和 → 量能累积
vol_sum = op.Ts_Sum(data.volume, 10)
```

---

##### `Ts_Mean(df, window)` — 时序滚动均值

```python
result = op.Ts_Mean(df, window)
```

对每只股票计算过去 `window` 天的**滚动均值**（等价于 `Ts_Sum / window`）。

```python
# 20日均价（MA20）
ma20 = op.Ts_Mean(data.close, 20)
# 价格相对均线偏离
factor_bias = op.Rank(data.close / ma20 - 1)
```

---

##### `Ts_Max(df, window)` — 时序滚动最大值

```python
result = op.Ts_Max(df, window)
```

对每只股票计算过去 `window` 天的**滚动最大值**。

```python
# 20日最高价
high20 = op.Ts_Max(data.high, 20)
```

---

##### `Ts_Min(df, window)` — 时序滚动最小值

```python
result = op.Ts_Min(df, window)
```

对每只股票计算过去 `window` 天的**滚动最小值**。

```python
# 20日最低价
low20 = op.Ts_Min(data.low, 20)
```

---

##### `Ts_Std(df, window)` — 时序滚动标准差

```python
result = op.Ts_Std(df, window)
```

对每只股票计算过去 `window` 天的**滚动标准差**（无偏估计，ddof=1）。

**返回值域：** ≥ 0，前 `window-1` 行为 NaN。

```python
# 低波动因子：过去20日收益率标准差
daily_ret = data.close.pct_change()
factor_lowvol = op.Rank(-op.Ts_Std(daily_ret, 20))
```

---

##### `Ts_Delta(df, period)` — 时序变化量

```python
result = op.Ts_Delta(df, period)
```

计算每只股票 `period` 天前的**价格变化量**：`df(t) - df(t - period)`。

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | DataFrame | 输入矩阵 |
| `period` | int | 回望天数 |

**返回值域：** 与输入同量纲（正=上涨，负=下跌），前 `period` 行为 NaN。

```python
# 5日价格变化量
delta5 = op.Ts_Delta(data.close, 5)
# 反转因子：近期跌幅越大，排名越靠前
factor_rev5 = op.Rank(-delta5)
```

---

##### `Ts_Delay(df, period)` — 时序延迟（平移）

```python
result = op.Ts_Delay(df, period)
```

将数据向后平移 `period` 天（等价于 `df.shift(period)`）。

**核心用途：** 防未来函数的显式延迟操作（`VectorEngine` 内 `delay=1` 参数在内部自动调用此算子）。

```python
# 一个月前的价格
close_21d_ago = op.Ts_Delay(data.close, 21)

# 跳过短期反转的月度动量：21日收益 - 5日收益
momentum = data.close / op.Ts_Delay(data.close, 21) - data.close / op.Ts_Delay(data.close, 5)
factor_mom = op.Rank(momentum)
```

---

##### `Ts_Rank(df, window)` — 时序分位排名

```python
result = op.Ts_Rank(df, window)   # 返回值域 [0, 1]
```

计算当前值在过去 `window` 天历史值中的**百分位排名**（从小到大的分位数）。

**返回值域：** [0, 1]，值越大表示当前值越接近历史高点。

> ⚠️ 性能提示：`Ts_Rank` 内部使用 `rolling.apply`（Python 层循环），对大矩阵（股票多、窗口大）运行较慢。建议 `window ≤ 60`，股票数 ≤ 500。

```python
# 价格处于近20日历史高位（突破类信号）
ts_rank_price = op.Ts_Rank(data.close, 20)
factor_ts_rank = op.Rank(ts_rank_price)
```

---

##### `Ts_Corr(df1, df2, window)` — 时序滚动相关系数

```python
result = op.Ts_Corr(df1, df2, window)   # 返回值域 [-1, 1]
```

计算两个矩阵在滚动窗口内的**逐列 Pearson 相关系数**。

| 参数 | 类型 | 说明 |
|------|------|------|
| `df1` | DataFrame | 第一个矩阵，形状须与 `df2` 相同 |
| `df2` | DataFrame | 第二个矩阵 |
| `window` | int | 滚动窗口大小 |

**返回值域：** [-1, 1]，正值表示同向变动，负值表示反向变动。

**经典用途：** 量价背离——`Ts_Corr(volume, close)` 为负表示**缩量上涨**（量价背离，通常为优质信号）。

```python
# 量价背离因子（10日量价相关系数）
vp_corr = op.Ts_Corr(data.volume, data.close, 10)
factor_vp = op.Neutralize(op.Rank(-vp_corr), data.industry)
```

---

## 截面类 (Cross-Sectional) (v1)

对每个时间截面（每一行）在股票维度做横向处理。

---

##### `Rank(df)` — 截面百分比排名

```python
result = op.Rank(df)   # 返回值域 [0, 1]
```

对每一行（截面）做**百分比排名**（`pct=True`），将原始因子值转换为在所有股票中的相对排名。

**返回值域：** [0, 1]，值越大表示该股票在当日截面中排名越靠前（值越大）。

> **推荐在任何因子输出前都包一层 `Rank`**，好处：
> 1. 消除量纲和极端值影响
> 2. 统一信号强度区间 [0, 1]
> 3. 提升跨期稳定性

```python
# 最常见用法
factor = op.Rank(op.Ts_Delta(data.close, 5))
```

---

##### `ZScore(df)` — 截面 Z-Score 标准化

```python
result = op.ZScore(df)   # 返回值域：理论上 (-∞, +∞)，实际约 [-3, 3]
```

对每一行（截面）做**Z-Score 标准化**：`(x - mean(x)) / std(x)`。

**返回值域：** 理论无界，实际约 [-3, 3]，均值为 0、标准差为 1。

- **> 0**：该股票当日因子值高于截面均值
- **< 0**：该股票当日因子值低于截面均值

> 与 `Rank` 的区别：`ZScore` 保留原始分布形状（离群值影响更大），`Rank` 变成均匀分布（对极端值更鲁棒）。

```python
# 标准化后再结合 Ts_Std
factor_zs = op.ZScore(op.Ts_Delta(data.close, 5))
```

---

##### `Scale(df, target_sum=1.0)` — 截面绝对值归一化

```python
result = op.Scale(df, target_sum=1.0)   # 每行绝对值之和 = target_sum
```

对每一行（截面）做**绝对值归一化**：`x / sum(|x|) × target_sum`。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `target_sum` | 1.0 | 目标绝对值之和，通常设为 1（做多等权组合）或 2（多空对称）|

**返回值域：** 保留正负号，每行绝对值之和等于 `target_sum`。

**核心用途：** 将多空因子值缩放为标准权重，便于比较不同因子的组合权重。

```python
factor_scaled = op.Scale(op.ZScore(raw_factor), target_sum=1.0)
```

---

## 特殊类 (Special) (v1)

跨时间和截面的复合处理算子。

---

##### `Decay_Linear(df, d)` — 线性衰减移动平均

```python
result = op.Decay_Linear(df, d)
```

对每只股票做**线性加权移动平均**：最近 `d` 天的因子值，按 `[d, d-1, ..., 1]` 线性加权，最近权重最大。

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | DataFrame | 输入矩阵（T × N） |
| `d` | int | 衰减窗口大小（天数） |

**核心用途：** 平滑因子信号、降低换手率。

```python
# 平滑5日动量信号（降换手率）
factor_smooth = op.Decay_Linear(op.Rank(op.Ts_Delta(data.close, 10)), d=5)
```

> 等价于 `VectorEngine` 中的 `decay` 参数（`VectorEngine` 内部自动调用此算子）。

---

##### `Neutralize(df, industry)` — 行业中性化

```python
result = op.Neutralize(df, industry)
```

对因子矩阵做**截面 OLS 行业中性化**：以行业虚拟变量为 X，因子值为 Y，取回归残差。

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | DataFrame | 因子矩阵（T × N） |
| `industry` | Series/DataFrame | 行业映射（股票 → 行业标签字符串，index=股票代码）|

**返回值域：** 与输入相同维度的残差矩阵，每个行业内均值接近 0。

```python
# 行业中性化：消除行业轮动对因子的干扰
factor_neut = op.Neutralize(
    op.Rank(-op.Ts_Corr(data.volume, data.close, 10)),
    data.industry
)
```

> 等价于 `VectorEngine` 中的 `industry` 参数（传入行业映射时内部自动调用此算子）。

---

## 🆕 时序统计算子 (v2)

v2.0 新增 4 个高阶时序统计算子，捕捉分布形态和序列依赖结构。

---

##### `Ts_Skew(df, window)` — 时序滚动偏度

```python
result = op.Ts_Skew(df, window)   # 返回值域：(-∞, +∞)，典型范围 [-3, 3]
```

对每只股票计算过去 `window` 天收益率的**滚动三阶矩偏度**（Fisher 定义）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | DataFrame | 输入矩阵（通常为日收益率 `close.pct_change()`） |
| `window` | int | 滚动窗口大小（建议 ≥ 20 保证统计稳定性） |

**返回值域：** 理论无界，典型范围 [-3, 3]。

- **> 0**（正偏）：分布右尾厚，存在极端上涨风险
- **< 0**（负偏）：分布左尾厚，存在极端下跌风险（投资者厌恶）

**因子含义：** 低负偏度（尾部下行风险小）的股票往往获得更高预期收益（作为风险补偿减少）。

```python
daily_ret = data.close.pct_change()
# 低负偏度因子（反转负偏度的负号 → 负偏度越小越好）
factor_skew = op.Rank(-op.Ts_Skew(daily_ret, 20))
```

---

##### `Ts_Kurt(df, window)` — 时序滚动峰度

```python
result = op.Ts_Kurt(df, window)   # 返回值域：理论 ≥ -2，正态分布 = 0
```

对每只股票计算过去 `window` 天的**滚动超额峰度**（Fisher 超额峰度，正态分布=0）。

- **> 0**（尖峰厚尾）：存在极端价格跳跃风险
- **< 0**（平坦分布）：收益分布较均匀，波动更稳定

```python
daily_ret = data.close.pct_change()
# 低峰度（价格跳跃风险小）→ 稳定股票
factor_kurt = op.Rank(-op.Ts_Kurt(daily_ret, 20))
```

---

##### `Ts_Autocorr(df, window, lag=1)` — 时序滚动自相关系数

```python
result = op.Ts_Autocorr(df, window, lag=1)   # 返回值域 [-1, 1]
```

对每只股票计算过去 `window` 天的 **lag 阶滚动自相关系数**（序列与自身平移 lag 期的 Pearson 相关）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window` | — | 滚动窗口大小 |
| `lag` | 1 | 自相关阶数（通常取 1） |

**返回值域：** [-1, 1]。

- **> 0**（正自相关）：价格/收益具有趋势性（动量特征）
- **< 0**（负自相关）：价格/收益具有反转性（均值回归特征）

```python
daily_ret = data.close.pct_change()
# 正自相关 → 动量
factor_ac = op.Rank(op.Ts_Autocorr(daily_ret, 20, lag=1))
# 负自相关 → 反转
factor_rev_ac = op.Rank(-op.Ts_Autocorr(daily_ret, 20, lag=1))
```

---

##### `Ts_Hurst(df, window)` — Hurst 指数

```python
result = op.Ts_Hurst(df, window)   # 返回值域 (0, 1)
```

基于 R/S 分析计算 Hurst 指数，衡量时间序列的**长记忆性**和趋势持续度。

**返回值域：** (0, 1)，计算需要足够长的窗口（建议 `window ≥ 100`）。

| 值域 | 含义 |
|------|------|
| H > 0.5 | 正持续性（趋势强，动量策略有效）|
| H ≈ 0.5 | 随机游走（无可利用规律）|
| H < 0.5 | 负持续性（均值回归，反转策略有效）|

```python
# 高 Hurst → 强趋势 → 配合动量信号
hurst = op.Ts_Hurst(data.close, window=120)
# Hurst > 0.5 的股票，用于动量策略筛选
factor_trend = op.Rank(hurst)
```

---

## 🆕 量价因子算子 (v2)

v2.0 新增 4 个量价因子算子，挖掘成交量与价格之间的信息关系。

---

##### `VWAP(close, volume, window)` — 成交量加权均价

```python
result = op.VWAP(close, volume, window)
```

计算滚动窗口内的**成交量加权平均价格（VWAP）**：`sum(close × volume) / sum(volume)`。

| 参数 | 类型 | 说明 |
|------|------|------|
| `close` | DataFrame | 收盘价矩阵 |
| `volume` | DataFrame | 成交量矩阵 |
| `window` | int | 滚动窗口大小（天数） |

**返回值域：** 与价格同量纲（元）。

> `VWAP` 是量价因子的核心基础算子，与 `VWAP_Bias`、`PVDeviation` 配合使用。

```python
vwap = op.VWAP(data.close, data.volume, window=10)
# 价格高于 VWAP → 强势；低于 VWAP → 弱势
price_vs_vwap = op.Rank(data.close / vwap - 1)
```

---

##### `VWAP_Bias(close, volume, window)` — VWAP 乖离率

```python
result = op.VWAP_Bias(close, volume, window)
```

衡量当前收盘价相对滚动 VWAP 的**百分比偏离**：`Close / VWAP(window) - 1`

| 参数 | 类型 | 说明 |
|------|------|------|
| `close` | DataFrame | 收盘价矩阵 |
| `volume` | DataFrame | 成交量矩阵 |
| `window` | int | 计算 VWAP 的滚动窗口大小 |

**返回值域：** 无界，典型范围 [-0.1, 0.1]（即 ±10%）。

- **> 0**：价格高于 VWAP，多头占优（可能相对超买）
- **< 0**：价格低于 VWAP，空头占优（潜在均值回归机会）

> 与 `PVDeviation` 的区别：`VWAP_Bias` 保留价格量纲（百分比偏离），`PVDeviation` 用价格标准差标准化（无量纲），两者信息互补。

```python
bias = op.VWAP_Bias(data.close, data.volume, window=10)
# 均值回归信号：大幅低于 VWAP → 短期反弹
factor_bias_rev = op.Rank(-bias)
# 趋势信号：持续正乖离 → 动量强
factor_bias_mom = op.Rank(bias)
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

## 🆕 动量因子算子 (v2)

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

## 🆕 技术指标算子 (v2)

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

## 🆕 缩量稳价因子 (V3)

V3 新增 4 个专注于**缩量稳价**（低成交量伴随价格稳定）现象的因子算子，捕捉筹码稳定、主力潜伏特征。

---

##### `VolSpike(volume, window)` — 成交量异动强度

```python
result = op.VolSpike(volume, window)   # 返回值域：[0, +∞)，典型范围 [0, 5]
```

衡量近期成交量相对历史均值的**放量倍数**，捕捉异常放量事件。

**计算逻辑：**
1. 计算 `window` 日滚动均值 `vol_ma`（基准量能）
2. 计算 `window` 日滚动标准差 `vol_std`（量能波动）
3. 输出 `(volume - vol_ma) / (vol_std + ε)`（量能 Z-Score 裁剪至 ≥ 0）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `volume` | DataFrame | — | 成交量矩阵（T × N） |
| `window` | int | 20 | 历史均量/标准差的滚动窗口（建议 10~30） |

**返回值域：** ≥ 0（负值被裁剪为 0）；典型范围 [0, 5]。

| 值 | 含义 |
|----|------|
| ≈ 0 | 成交量正常或低于均值（缩量） |
| ≈ 1 | 轻度放量（约均值+1倍标准差） |
| ≥ 2 | 显著放量（统计异常，值得关注） |

```python
# 异动放量 → 可能是主力介入信号
vol_spike = op.VolSpike(data.volume, window=20)
# 单独用于放量突破策略
factor_spike = op.Rank(vol_spike)
```

---

##### `PriceVarShrink(close, volume, price_window, vol_window)` — 缩量价格波动收缩

```python
result = op.PriceVarShrink(close, volume,
                            price_window=10, vol_window=20)
```

捕捉**缩量同时伴随价格波动收窄**的特征，识别震荡整理、筹码锁定阶段。

**计算逻辑：**
1. 计算 `price_window` 日价格滚动标准差 `price_std`（价格波动）
2. 计算 `vol_window` 日成交量滚动均值 `vol_ma`，取负值 `-vol_ma`（缩量信号）
3. 对两个维度分别截面 Rank，然后**等权合成**（缩量 + 价格波动收窄）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `close` | DataFrame | — | 收盘价矩阵 |
| `volume` | DataFrame | — | 成交量矩阵 |
| `price_window` | int | 10 | 价格标准差计算窗口 |
| `vol_window` | int | 20 | 成交量均值计算窗口 |

**返回值域：** [0, 1]（两个 Rank 的等权平均）。

```python
pvs = op.PriceVarShrink(data.close, data.volume,
                         price_window=10, vol_window=20)
# 缩量 + 波动收窄 → 筹码集中，潜在突破信号
factor_pvs = op.Rank(pvs)
```

---

##### `PriceMeanShrink(close, volume, price_window, vol_window)` — 缩量价格趋势平坦

```python
result = op.PriceMeanShrink(close, volume,
                             price_window=10, vol_window=20)
```

捕捉**缩量同时伴随价格滚动均值变化放缓（趋势平坦）**的特征，识别价格处于蓄势横盘阶段。

**计算逻辑：**
1. 计算 `price_window` 日均价变动幅度 `|ma_diff|`（均线斜率绝对值）
2. 计算 `vol_window` 日成交量均值，取负值 `-vol_ma`（缩量信号）
3. 对两个维度分别截面 Rank，然后**等权合成**（缩量 + 均价趋势平坦）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `close` | DataFrame | — | 收盘价矩阵 |
| `volume` | DataFrame | — | 成交量矩阵 |
| `price_window` | int | 10 | 均价变动计算窗口 |
| `vol_window` | int | 20 | 成交量均值计算窗口 |

**返回值域：** [0, 1]（两个 Rank 的等权平均）。

```python
pms = op.PriceMeanShrink(data.close, data.volume,
                          price_window=10, vol_window=20)
# 缩量 + 均线走平 → 横盘蓄势
factor_pms = op.Rank(pms)
```

---

##### `VolSpikeStablePrice(close, volume, price_window, vol_window)` — 缩量稳价综合评分

```python
result = op.VolSpikeStablePrice(close, volume,
                                 price_window=10, vol_window=20)
```

综合 `PriceVarShrink` 与 `PriceMeanShrink` 两个维度，生成更全面的**缩量稳价**评分，是 V3 的核心合成因子。

**计算逻辑：** `0.5 × PriceVarShrink + 0.5 × PriceMeanShrink`（等权合成）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `close` | DataFrame | — | 收盘价矩阵 |
| `volume` | DataFrame | — | 成交量矩阵 |
| `price_window` | int | 10 | 价格窗口（同时用于方差和均值计算） |
| `vol_window` | int | 20 | 成交量均值计算窗口 |

**返回值域：** [0, 1]。

**投资含义：**
- 高分股票：成交量持续低于历史均值 + 价格波动率低 + 均线走平
- 这类股票处于主力吸筹/筹码锁定阶段，一旦成交量放大（`VolSpike` 高分）则可能触发突破

```python
# 核心合成因子（推荐用法）
vss = op.VolSpikeStablePrice(data.close, data.volume,
                              price_window=10, vol_window=20)
factor_vss = op.Rank(vss)

# 完整策略：缩量稳价选股 + 行业中性化
factor_final = op.Neutralize(
    op.Rank(op.VolSpikeStablePrice(data.close, data.volume)),
    data.industry
)
```

---

##### `Bollinger_Outlier_Frequency(close, period=30, k=3.5, lookback_window=60)` — 布林带极端突破频率

```python
result = op.Bollinger_Outlier_Frequency(close, period=30, k=3.5, lookback_window=60)
# 返回值域：[0, 1]
```

**核心逻辑：**
1. 计算 `period` 日简单移动均线（中轨 MB）与滚动标准差（std）
2. 构建极宽布林带：上轨 = MB + k × std，下轨 = MB − k × std（默认 k=3.5，理论突破率 < 0.05%）
3. 判断每日是否"离群"：close > 上轨 **或** close < 下轨，记为 1，否则记为 0
4. 在 `lookback_window` 天内对离群标记求滚动均值，得到突破频率

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `close` | DataFrame | — | **必填**。收盘价矩阵 (T × N) |
| `period` | int | 30 | 布林带均线与标准差的计算窗口（天数） |
| `k` | float | 3.5 | 标准差倍数，值越大带越宽，突破越稀有；推荐范围 [2.0, 4.0] |
| `lookback_window` | int | 60 | 统计突破频率的回测窗口（天数） |

**返回值域：** [0, 1]，前 `period + lookback_window - 2` 行为 NaN。

| 值 | 含义 |
|----|------|
| 0 | 过去 lookback_window 天内从未突破布林带 |
| 0.05～0.15 | 存在间歇性价格突破，趋势性较强 |
| > 0.2 | 频繁突破，价格行为高度异常或处于强趋势中 |

> **k 值选取建议：** k=2.0（常规布林带，突破率 ~4.5%），k=2.5（突破率 ~1.2%），k=3.0（突破率 ~0.27%），k=3.5（突破率 ~0.047%，推荐用于筛选极端行情）。

```python
# 基础用法：计算布林带突破频率
f_bof = op.Bollinger_Outlier_Frequency(data.close, period=30, k=3.5, lookback_window=60)

# 截面排名后送入回测（推荐）
factor_bof = op.Rank(f_bof)
result = VectorEngine(
    factor=factor_bof, close=data.close,
    is_suspended=data.is_suspended, is_limit=data.is_limit,
    rebalance_freq=5, top_n=30,
).run()

# 不同 k 值对比（稀有程度不同）
f_k20 = op.Bollinger_Outlier_Frequency(data.close, k=2.0, lookback_window=60)  # 常规
f_k35 = op.Bollinger_Outlier_Frequency(data.close, k=3.5, lookback_window=60)  # 极端（推荐）
```

> ⚠️ **NaN 处理：** 当价格在整个 `period` 窗口内完全不变（std=0，例如长期停牌后首日）时，std 自动置为 NaN，避免虚假突破信号。

---

## 算子速查表

| 算子 | 类别 | 版本 | 核心输入 | 返回值域 | 典型窗口 |
|------|------|------|----------|----------|----------|
| `Ts_Sum` | 时序 | v1 | df | 同量纲 | 5~20 |
| `Ts_Mean` | 时序 | v1 | df | 同量纲 | 5~60 |
| `Ts_Max` | 时序 | v1 | df | 同量纲 | 10~60 |
| `Ts_Min` | 时序 | v1 | df | 同量纲 | 10~60 |
| `Ts_Std` | 时序 | v1 | df | ≥0 | 10~60 |
| `Ts_Delta` | 时序 | v1 | df | 同量纲 | 5~21 |
| `Ts_Delay` | 时序 | v1 | df | 同量纲 | 1~252 |
| `Ts_Rank` | 时序 | v1 | df | [0,1] | 10~60 |
| `Ts_Corr` | 时序 | v1 | df1, df2 | [-1,1] | 5~30 |
| `Rank` | 截面 | v1 | df | [0,1] | — |
| `ZScore` | 截面 | v1 | df | ~[-3,3] | — |
| `Scale` | 截面 | v1 | df | ±bounded | — |
| `Decay_Linear` | 特殊 | v1 | df | 同量纲 | 3~10 |
| `Neutralize` | 特殊 | v1 | df, industry | 残差 | — |
| `Ts_Skew` | 时序统计 | v2 | df | ~[-3,3] | 20~60 |
| `Ts_Kurt` | 时序统计 | v2 | df | ≥-2 | 20~60 |
| `Ts_Autocorr` | 时序统计 | v2 | df | [-1,1] | 20~60 |
| `Ts_Hurst` | 时序统计 | v2 | df | (0,1) | ≥100 |
| `VWAP` | 量价 | v2 | close, vol | 价格量纲 | 5~20 |
| `VWAP_Bias` | 量价 | v2 | close, vol | ~[-0.1,0.1] | 5~20 |
| `PVDeviation` | 量价 | v2 | close, vol | 标准化 | 5~20 |
| `Amihud` | 量价 | v2 | close, vol | ≥0 | 10~30 |
| `RiskAdjMomentum` | 动量 | v2 | close | 无界 | 10~60 |
| `PricePathQuality` | 动量 | v2 | close | [-1,1] | 10~30 |
| `RangeBreakout` | 动量 | v2 | close,h,l | [0,1] | 10~60 |
| `RSI` | 技术 | v2 | close | [0,100] | 14 |
| `KDJ` | 技术 | v2 | close,h,l | ~[0,100] | 9 |
| `MACD` | 技术 | v2 | close | 无界 | 12/26/9 |
| `VolSpike` | 缩量稳价 | V3 | volume | [0,+∞) | 10~30 |
| `PriceVarShrink` | 缩量稳价 | V3 | close, vol | [0,1] | 10/20 |
| `PriceMeanShrink` | 缩量稳价 | V3 | close, vol | [0,1] | 10/20 |
| `VolSpikeStablePrice` | 缩量稳价 | V3 | close, vol | [0,1] | 10/20 |
| `Bollinger_Outlier_Frequency` | 布林带异动 | V3 | close | [0,1] | period=30, lookback=60 |

---

*返回主文档：[README.md](./README.md)*
