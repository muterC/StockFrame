"""
Report — 回测可视化报告生成模块
=================================
基于 Matplotlib + Seaborn 生成专业的 6 子图回测分析报告：

布局（2 行 × 3 列）
-------------------
[0,0] 净值曲线       —— 策略净值 vs 基准，最大回撤阴影
[0,1] 月度收益热力图 —— 行=年，列=月，红绿配色
[0,2] IC 时序柱状图  —— 正/负 IC 颜色区分，IC_Mean 虚线
[1,0] 每日收益分布   —— 直方图 + KDE + 正态分布对比
[1,1] 换手率序列     —— 折线图 + 均值虚线 + 调仓日标记
[1,2] IC 分布        —— 直方图 + 正态拟合 + 胜率标注

使用方式
--------
>>> result.plot()                       # 弹窗展示
>>> result.plot(save_path="report.png") # 保存到文件
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from scipy import stats as sp_stats

if TYPE_CHECKING:
    from quant_alpha_engine.backtest.vector_engine import BacktestResult

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 中文字体配置：自动寻找系统中可用的 CJK 字体
# ---------------------------------------------------------------------------

def _setup_chinese_font() -> None:
    """
    自动配置 Matplotlib 中文字体，兼容 Windows / macOS / Linux。

    优先顺序（Windows 优先，其次 macOS，再次 Linux 常见字体）：
    微软雅黑 → 黑体 → 思源黑体 → PingFang → Hiragino → Noto → WenQuanYi

    若均不可用，回退到英文显示（不崩溃，但中文显示为方块）。
    """
    from matplotlib import font_manager

    CJK_CANDIDATES = [
        # Windows
        "Microsoft YaHei",     # 微软雅黑
        "SimHei",              # 黑体
        "SimSun",              # 宋体
        "KaiTi",               # 楷体
        "FangSong",            # 仿宋
        # macOS
        "PingFang SC",
        "Hiragino Sans GB",
        "STHeiti",
        "STSong",
        # Linux / 开源
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
        "Droid Sans Fallback",
    ]

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in CJK_CANDIDATES:
        if name in available:
            chosen = name
            break

    if chosen:
        plt.rcParams["font.family"] = "sans-serif"
        # 将选中的中文字体放在最前面，后跟通用 sans-serif 备选
        existing = plt.rcParams.get("font.sans-serif", [])
        if not isinstance(existing, list):
            existing = list(existing)
        # 去重并插到最前
        new_list = [chosen] + [f for f in existing if f != chosen]
        plt.rcParams["font.sans-serif"] = new_list
    # 修复负号乱码问题（无论是否找到中文字体都需要设置）
    plt.rcParams["axes.unicode_minus"] = False


# 模块加载时执行一次
_setup_chinese_font()


# ---------------------------------------------------------------------------
# 全局样式配置
# ---------------------------------------------------------------------------
_COLORS = {
    "strategy":   "#FF4B4B",   # 策略净值 — 亮红
    "benchmark":  "#6C7A89",   # 基准      — 灰蓝
    "drawdown":   "#FF4B4B",   # 回撤阴影
    "ic_pos":     "#2ECC71",   # 正 IC     — 绿
    "ic_neg":     "#E74C3C",   # 负 IC     — 红
    "ic_mean":    "#F39C12",   # IC 均值线 — 橙
    "turnover":   "#3498DB",   # 换手率    — 蓝
    "rebal":      "#95A5A6",   # 调仓日标记
    "dist":       "#3498DB",   # 收益分布  — 蓝
    "normal_fit": "#E67E22",   # 正态拟合  — 橙
    "zero_line":  "#FFFFFF",   # 零线
    "grid":       "#2F3640",
    "text":       "#ECF0F1",
    "bg":         "#1A1A2E",
    "ax_bg":      "#16213E",
    "spine":      "#0F3460",
}


class Report:
    """回测可视化报告生成器（全静态方法）。"""

    @staticmethod
    def plot(
        result: "BacktestResult",
        save_path: Optional[str] = None,
        benchmark_seed: int = 2024,
    ) -> None:
        """
        生成完整的 6 子图回测分析报告并展示/保存。

        Parameters
        ----------
        result         : BacktestResult 对象
        save_path      : 若指定，保存为 PNG 文件（dpi=150）；否则弹窗展示
        benchmark_seed : 生成基准序列的随机种子
        """
        # ------------------------------------------------------------------
        # 0. 准备数据
        # ------------------------------------------------------------------
        nav      = result.nav.dropna()
        ret      = result.daily_returns.dropna()
        turnover = result.turnover.dropna()
        ic       = result.ic_series.dropna()
        metrics  = result.metrics
        rebal_dates = result.rebalance_dates

        # 生成模拟基准（随机游走，年化波动 ~18%）
        rng = np.random.default_rng(benchmark_seed)
        bm_ret = rng.normal(0.00025, 0.012, size=len(nav))
        bm_nav = pd.Series(
            np.cumprod(1 + bm_ret),
            index=nav.index,
            name="benchmark"
        )
        bm_nav = bm_nav / bm_nav.iloc[0]

        # 月度收益（用于热力图）
        monthly_ret = Report._calc_monthly_returns(ret)

        # ------------------------------------------------------------------
        # 1. 构建画布
        # ------------------------------------------------------------------
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(22, 14), facecolor=_COLORS["bg"])
        fig.suptitle(
            "QuantAlpha Engine — 因子回测分析报告",
            fontsize=16, fontweight="bold",
            color=_COLORS["text"], y=0.98,
            fontfamily="sans-serif",
        )

        gs = gridspec.GridSpec(
            2, 3,
            figure=fig,
            hspace=0.40,
            wspace=0.32,
            left=0.06, right=0.97,
            top=0.93, bottom=0.07,
        )

        axes = {
            "nav":      fig.add_subplot(gs[0, 0]),
            "heatmap":  fig.add_subplot(gs[0, 1]),
            "ic_bar":   fig.add_subplot(gs[0, 2]),
            "ret_dist": fig.add_subplot(gs[1, 0]),
            "turnover": fig.add_subplot(gs[1, 1]),
            "ic_dist":  fig.add_subplot(gs[1, 2]),
        }

        for ax in axes.values():
            ax.set_facecolor(_COLORS["ax_bg"])
            for spine in ax.spines.values():
                spine.set_color(_COLORS["spine"])
            ax.tick_params(colors=_COLORS["text"], labelsize=8)
            ax.xaxis.label.set_color(_COLORS["text"])
            ax.yaxis.label.set_color(_COLORS["text"])
            ax.title.set_color(_COLORS["text"])

        # ------------------------------------------------------------------
        # 2. 子图绘制
        # ------------------------------------------------------------------
        Report._plot_nav(axes["nav"],  nav, bm_nav, metrics)
        Report._plot_heatmap(axes["heatmap"], monthly_ret)
        Report._plot_ic_bar(axes["ic_bar"],   ic, metrics)
        Report._plot_ret_dist(axes["ret_dist"], ret, metrics)
        Report._plot_turnover(axes["turnover"], turnover, rebal_dates, metrics)
        Report._plot_ic_dist(axes["ic_dist"],   ic, metrics)

        # ------------------------------------------------------------------
        # 3. 添加顶部指标摘要条
        # ------------------------------------------------------------------
        Report._add_metrics_strip(fig, metrics)

        # ------------------------------------------------------------------
        # 4. 保存或展示
        # ------------------------------------------------------------------
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=_COLORS["bg"])
            print(f"[Report] 报告已保存至: {save_path}")
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

    # ==================================================================
    # 子图：净值曲线
    # ==================================================================

    @staticmethod
    def _plot_nav(
        ax: plt.Axes,
        nav: pd.Series,
        bm_nav: pd.Series,
        metrics: dict,
    ) -> None:
        """净值曲线 + 最大回撤阴影。"""
        ax.set_title("📈 策略净值曲线", fontsize=10, pad=8)

        # 绘制基准
        ax.plot(bm_nav.index, bm_nav.values,
                color=_COLORS["benchmark"], linewidth=1.0,
                linestyle="--", alpha=0.7, label="基准(模拟)")

        # 绘制策略净值
        ax.plot(nav.index, nav.values,
                color=_COLORS["strategy"], linewidth=1.8,
                label=f"策略", zorder=5)

        # 最大回撤阴影
        rolling_max = nav.cummax()
        drawdown    = (nav - rolling_max) / rolling_max
        mdd_end_idx = drawdown.idxmin()
        # 找回撤起点（从 mdd_end 往前找最大值）
        mdd_start_idx = rolling_max[:mdd_end_idx].idxmax()

        ax.fill_between(
            nav.index,
            nav.values,
            rolling_max.values,
            where=(drawdown < 0),
            alpha=0.15,
            color=_COLORS["drawdown"],
            label=f"回撤区间",
        )

        # 标注最大回撤
        mdd_val = metrics.get("最大回撤", 0)
        ax.annotate(
            f"MaxDD: {mdd_val * 100:.1f}%",
            xy=(mdd_end_idx, nav[mdd_end_idx]),
            xytext=(0.60, 0.12),
            textcoords="axes fraction",
            fontsize=8, color=_COLORS["drawdown"],
            arrowprops=dict(arrowstyle="->", color=_COLORS["drawdown"], lw=1.0),
        )

        # 零基准线
        ax.axhline(1.0, color=_COLORS["zero_line"], linewidth=0.6, alpha=0.5)

        ax.set_xlabel("日期", fontsize=8)
        ax.set_ylabel("净值", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=7, loc="upper left",
                  facecolor=_COLORS["bg"], edgecolor=_COLORS["spine"],
                  labelcolor=_COLORS["text"])
        ax.grid(True, color=_COLORS["grid"], alpha=0.5, linewidth=0.5)
        ax.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter("%Y-%m")
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # ==================================================================
    # 子图：月度收益热力图
    # ==================================================================

    @staticmethod
    def _plot_heatmap(ax: plt.Axes, monthly_ret: pd.DataFrame) -> None:
        """月度收益热力图（行=年，列=月）。"""
        ax.set_title("🗓️ 月度收益热力图", fontsize=10, pad=8)

        if monthly_ret is None or monthly_ret.empty:
            ax.text(0.5, 0.5, "数据不足", ha="center", va="center",
                    color=_COLORS["text"], transform=ax.transAxes)
            return

        # 构建注释标签（百分比）
        annot = monthly_ret.applymap(
            lambda v: f"{v*100:.1f}%" if not np.isnan(v) else ""
        )

        if HAS_SEABORN:
            sns.heatmap(
                monthly_ret * 100,
                ax=ax,
                annot=annot,
                fmt="",
                cmap="RdYlGn",
                center=0,
                linewidths=0.5,
                linecolor=_COLORS["bg"],
                cbar_kws={"shrink": 0.8, "format": "%.1f%%"},
                annot_kws={"size": 7},
            )
            ax.collections[0].colorbar.ax.tick_params(
                colors=_COLORS["text"], labelsize=7
            )
        else:
            im = ax.imshow(
                monthly_ret.values * 100,
                cmap="RdYlGn", aspect="auto",
                vmin=monthly_ret.values.min() * 100,
                vmax=monthly_ret.values.max() * 100,
            )
            ax.set_xticks(range(len(monthly_ret.columns)))
            ax.set_xticklabels(monthly_ret.columns, fontsize=7)
            ax.set_yticks(range(len(monthly_ret.index)))
            ax.set_yticklabels(monthly_ret.index, fontsize=7)

        month_labels = ["1月","2月","3月","4月","5月","6月",
                         "7月","8月","9月","10月","11月","12月"]
        valid_months = [month_labels[m-1] for m in monthly_ret.columns]
        ax.set_xticklabels(valid_months, fontsize=7, color=_COLORS["text"])
        ax.set_yticklabels(monthly_ret.index, fontsize=7,
                           color=_COLORS["text"], rotation=0)
        ax.set_xlabel("月份", fontsize=8)
        ax.set_ylabel("年份", fontsize=8)

    # ==================================================================
    # 子图：IC 时序柱状图
    # ==================================================================

    @staticmethod
    def _plot_ic_bar(
        ax: plt.Axes,
        ic: pd.Series,
        metrics: dict,
    ) -> None:
        """IC 时序柱状图，正IC绿色负IC红色。"""
        ax.set_title("📊 IC 时序", fontsize=10, pad=8)

        if len(ic) == 0:
            ax.text(0.5, 0.5, "IC 数据不足", ha="center", va="center",
                    color=_COLORS["text"], transform=ax.transAxes)
            return

        colors = [_COLORS["ic_pos"] if v >= 0 else _COLORS["ic_neg"]
                  for v in ic.values]

        ax.bar(ic.index, ic.values, color=colors, alpha=0.75, width=1.5)

        # IC 均值线
        ic_mean = metrics.get("IC_Mean", np.nan)
        if not np.isnan(ic_mean):
            ax.axhline(
                ic_mean,
                color=_COLORS["ic_mean"],
                linewidth=1.5,
                linestyle="--",
                label=f"IC均值={ic_mean:.4f}",
            )

        ax.axhline(0, color=_COLORS["zero_line"], linewidth=0.6, alpha=0.5)

        # 标注 ICIR
        icir = metrics.get("ICIR", np.nan)
        if not np.isnan(icir):
            ax.text(
                0.02, 0.95,
                f"ICIR = {icir:.3f}",
                transform=ax.transAxes,
                fontsize=8, color=_COLORS["ic_mean"],
                va="top",
            )

        ax.set_xlabel("日期", fontsize=8)
        ax.set_ylabel("IC", fontsize=8)
        ax.legend(fontsize=7, facecolor=_COLORS["bg"],
                  edgecolor=_COLORS["spine"], labelcolor=_COLORS["text"])
        ax.grid(True, color=_COLORS["grid"], alpha=0.5, linewidth=0.5)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # ==================================================================
    # 子图：每日收益分布
    # ==================================================================

    @staticmethod
    def _plot_ret_dist(
        ax: plt.Axes,
        ret: pd.Series,
        metrics: dict,
    ) -> None:
        """日收益率分布直方图 + KDE + 正态对比。"""
        ax.set_title("📉 日收益率分布", fontsize=10, pad=8)

        clean = ret.dropna()
        if len(clean) < 10:
            ax.text(0.5, 0.5, "数据不足", ha="center", va="center",
                    color=_COLORS["text"], transform=ax.transAxes)
            return

        # 直方图
        n_bins = min(80, max(20, len(clean) // 10))
        ax.hist(
            clean * 100,
            bins=n_bins,
            color=_COLORS["dist"],
            alpha=0.55,
            density=True,
            label="实际分布",
            edgecolor="none",
        )

        # KDE（如果有 seaborn）
        x_range = np.linspace(clean.min() * 100, clean.max() * 100, 300)
        if len(clean) > 20:
            kde = sp_stats.gaussian_kde(clean * 100)
            ax.plot(x_range, kde(x_range),
                    color=_COLORS["dist"], linewidth=1.8, label="KDE")

        # 正态分布拟合
        mu_fit  = clean.mean() * 100
        std_fit = clean.std()  * 100
        ax.plot(
            x_range,
            sp_stats.norm.pdf(x_range, mu_fit, std_fit),
            color=_COLORS["normal_fit"],
            linewidth=1.5,
            linestyle="--",
            label="正态拟合",
        )

        ax.axvline(0, color=_COLORS["zero_line"], linewidth=0.8, alpha=0.6)
        ax.axvline(
            mu_fit,
            color=_COLORS["strategy"],
            linewidth=1.2,
            linestyle=":",
            label=f"均值={mu_fit:.3f}%",
        )

        ax.set_xlabel("日收益率 (%)", fontsize=8)
        ax.set_ylabel("概率密度", fontsize=8)
        ax.legend(fontsize=7, facecolor=_COLORS["bg"],
                  edgecolor=_COLORS["spine"], labelcolor=_COLORS["text"])
        ax.grid(True, color=_COLORS["grid"], alpha=0.5, linewidth=0.5)

    # ==================================================================
    # 子图：换手率序列
    # ==================================================================

    @staticmethod
    def _plot_turnover(
        ax: plt.Axes,
        turnover: pd.Series,
        rebal_dates: list,
        metrics: dict,
    ) -> None:
        """换手率序列折线图，标注调仓日。"""
        ax.set_title("🔄 换手率序列", fontsize=10, pad=8)

        nonzero_to = turnover[turnover > 1e-6]
        if len(nonzero_to) == 0:
            ax.text(0.5, 0.5, "无换手数据", ha="center", va="center",
                    color=_COLORS["text"], transform=ax.transAxes)
            return

        ax.plot(
            nonzero_to.index,
            nonzero_to.values * 100,
            color=_COLORS["turnover"],
            linewidth=0.8,
            alpha=0.8,
            label="日换手率",
        )

        # 均值线
        avg_to = metrics.get("日均换手率", np.nan)
        if not np.isnan(avg_to):
            ax.axhline(
                avg_to * 100,
                color=_COLORS["normal_fit"],
                linewidth=1.5,
                linestyle="--",
                label=f"均值={avg_to*100:.1f}%",
            )

        # 标记调仓日（垂直虚线，降低 alpha 避免遮挡）
        to_dates_set = set(nonzero_to.index)
        for d in rebal_dates[:50]:  # 最多标 50 条，避免拥挤
            if d in to_dates_set:
                ax.axvline(d, color=_COLORS["rebal"],
                           linewidth=0.3, alpha=0.3)

        ax.set_xlabel("日期", fontsize=8)
        ax.set_ylabel("换手率 (%)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.legend(fontsize=7, facecolor=_COLORS["bg"],
                  edgecolor=_COLORS["spine"], labelcolor=_COLORS["text"])
        ax.grid(True, color=_COLORS["grid"], alpha=0.5, linewidth=0.5)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # ==================================================================
    # 子图：IC 分布
    # ==================================================================

    @staticmethod
    def _plot_ic_dist(
        ax: plt.Axes,
        ic: pd.Series,
        metrics: dict,
    ) -> None:
        """IC 分布直方图 + 正态拟合 + 胜率标注。"""
        ax.set_title("📐 IC 分布", fontsize=10, pad=8)

        clean = ic.dropna()
        if len(clean) < 5:
            ax.text(0.5, 0.5, "IC 数据不足", ha="center", va="center",
                    color=_COLORS["text"], transform=ax.transAxes)
            return

        n_bins = min(50, max(15, len(clean) // 8))
        ax.hist(
            clean.values,
            bins=n_bins,
            color=_COLORS["ic_pos"],
            alpha=0.55,
            density=True,
            edgecolor="none",
            label="IC 分布",
        )

        # 正态拟合
        x_range = np.linspace(clean.min(), clean.max(), 300)
        mu_ic  = clean.mean()
        std_ic = clean.std()
        ax.plot(
            x_range,
            sp_stats.norm.pdf(x_range, mu_ic, std_ic),
            color=_COLORS["normal_fit"],
            linewidth=2.0,
            label="正态拟合",
        )

        ax.axvline(0, color=_COLORS["zero_line"], linewidth=0.8, alpha=0.6)
        ax.axvline(
            mu_ic,
            color=_COLORS["ic_mean"],
            linewidth=1.5,
            linestyle="--",
            label=f"均值={mu_ic:.4f}",
        )

        # 胜率标注
        win_rate = metrics.get("IC_胜率", np.nan)
        if not np.isnan(win_rate):
            ax.text(
                0.02, 0.95,
                f"IC胜率: {win_rate*100:.1f}%",
                transform=ax.transAxes,
                fontsize=8, color=_COLORS["text"], va="top",
            )

        ax.set_xlabel("IC 值", fontsize=8)
        ax.set_ylabel("概率密度", fontsize=8)
        ax.legend(fontsize=7, facecolor=_COLORS["bg"],
                  edgecolor=_COLORS["spine"], labelcolor=_COLORS["text"])
        ax.grid(True, color=_COLORS["grid"], alpha=0.5, linewidth=0.5)

    # ==================================================================
    # 顶部指标摘要条
    # ==================================================================

    @staticmethod
    def _add_metrics_strip(fig: plt.Figure, metrics: dict) -> None:
        """在图表顶部添加关键指标摘要文字行。"""

        def _pct(key):
            v = metrics.get(key, np.nan)
            if np.isnan(v):
                return "N/A"
            return f"{v * 100:.2f}%"

        def _num(key, fmt=".4f"):
            v = metrics.get(key, np.nan)
            if np.isnan(v):
                return "N/A"
            return f"{v:{fmt}}"

        strip_items = [
            f"年化收益: {_pct('年化收益率')}",
            f"Sharpe: {_num('Sharpe_Ratio', '.3f')}",
            f"MaxDD: {_pct('最大回撤')}",
            f"IC均值: {_num('IC_Mean')}",
            f"ICIR: {_num('ICIR', '.3f')}",
            f"IC胜率: {_pct('IC_胜率')}",
            f"日换手: {_pct('日均换手率')}",
            f"Fitness: {_num('Fitness', '.3f')}",
        ]
        strip_text = "   |   ".join(strip_items)

        fig.text(
            0.5, 0.955,
            strip_text,
            ha="center", va="center",
            fontsize=8.5,
            color="#F8F9FA",
            bbox=dict(
                facecolor="#0F3460",
                edgecolor="#3498DB",
                boxstyle="round,pad=0.3",
                alpha=0.8,
            ),
        )

    # ==================================================================
    # 辅助方法
    # ==================================================================

    @staticmethod
    def _calc_monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
        """
        将日收益率聚合为月度收益率矩阵（行=年，列=月）。

        月度收益 = prod(1 + daily_ret) - 1
        """
        if len(daily_returns) == 0:
            return pd.DataFrame()

        monthly = daily_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly.index = pd.PeriodIndex(monthly.index, freq="M")

        # 透视为 年×月 矩阵
        df = pd.DataFrame({
            "year":  monthly.index.year,
            "month": monthly.index.month,
            "ret":   monthly.values,
        })

        pivot = df.pivot(index="year", columns="month", values="ret")
        pivot.columns.name = "month"
        pivot.index.name   = "year"
        return pivot
