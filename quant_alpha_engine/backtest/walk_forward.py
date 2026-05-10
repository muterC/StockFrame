"""
WalkForward — 滚动样本外验证
================================
把整段历史切成「训练窗 + 测试窗」滑动序列，每个 OOS 窗口独立跑回测，
输出每窗 Sharpe / IC / 累计净值，用于评估策略稳定性与过拟合风险。

时序约定（无未来函数）
----------------------
    ─────────── train_window ───────────┃─── test_window ───┃ ...
    [t0 ............................. t1)[t1 .............. t2) ...
                                        ↑
                                    OOS 评估区间（仅此段计入）

每滑动一步，区间整体平移 step 个交易日。

典型用法
--------
>>> wf = WalkForward(
...     factor_fn      = my_factor,        # 给定 close 等数据，返回因子矩阵
...     close          = close,
...     is_suspended   = is_susp,
...     is_limit       = is_limit,
...     train_window   = 252,
...     test_window    = 63,
...     step           = 21,
...     backtest_kwargs= dict(top_n=30, rebalance_freq=5, delay=1),
... )
>>> report = wf.run()
>>> print(report.summary)            # 每窗 Sharpe/IC/MaxDD 表
>>> report.oos_sharpe_mean           # OOS 平均 Sharpe
>>> report.equity_curve.plot()       # 全 OOS 拼接的等权净值
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from quant_alpha_engine.backtest.vector_engine import VectorEngine, BacktestResult
from quant_alpha_engine.backtest.performance import Performance


# ===========================================================================
# 结果容器
# ===========================================================================

@dataclass
class WalkForwardReport:
    """
    Walk-Forward 报告。

    Attributes
    ----------
    windows         : 每个窗口的 (train_start, train_end, test_start, test_end)
    fold_metrics    : list[dict]，每个窗口的 OOS 指标
    summary         : 每窗指标拼成的 DataFrame（按 test_start 排序）
    equity_curve    : 把所有 OOS 窗口的 daily_returns 拼接得到的整体净值
    oos_sharpe_mean : OOS Sharpe 均值
    oos_sharpe_std  : OOS Sharpe 标准差（稳定性）
    oos_ic_mean     : OOS 平均 IC
    """
    windows:         List[tuple]
    fold_metrics:    List[dict]
    summary:         pd.DataFrame
    equity_curve:    pd.Series
    oos_sharpe_mean: float
    oos_sharpe_std:  float
    oos_ic_mean:     float
    fold_results:    List[BacktestResult] = field(default_factory=list)

    def print_summary(self) -> None:
        print("=" * 60)
        print(f"Walk-Forward OOS 汇总  (folds={len(self.fold_metrics)})")
        print("=" * 60)
        print(f"  OOS Sharpe   均值 = {self.oos_sharpe_mean:+.3f}")
        print(f"  OOS Sharpe   标差 = {self.oos_sharpe_std:+.3f}")
        print(f"  OOS IC       均值 = {self.oos_ic_mean:+.4f}")
        print(f"  OOS Sharpe   胜率 = "
              f"{(self.summary['Sharpe_Ratio'] > 0).mean() * 100:.1f}%")
        print("=" * 60)
        print(self.summary.to_string())


# ===========================================================================
# WalkForward 引擎
# ===========================================================================

class WalkForward:
    """
    滚动样本外验证。

    参数说明
    --------
    factor_fn : Callable -> pd.DataFrame
        因子工厂函数，签名约定为 ``factor_fn(close, **factor_kwargs) -> factor_df``。
        如果你的因子需要其它输入（volume、high、low 等），通过 ``factor_kwargs`` 传。
        框架会把传入的 close 自动按训练窗起点切片，让因子计算只看到「截至训练窗末」
        的数据，OOS 评估区间则单独切给 VectorEngine。
    close, is_suspended, is_limit : T×N 全样本数据矩阵
    train_window  : 训练窗口长度（交易日）。仅用于决定 OOS 评估的起点；
                    本框架的因子算子大多是滚动型，无需显式 fit，因此 train_window
                    主要充当「冷启动 / 防 look-ahead」的预热长度。
    test_window   : 每个 OOS 测试窗口长度（交易日）
    step          : 每次滑动步长（一般 = test_window 实现不重叠 OOS）
    factor_kwargs : 传给 factor_fn 的额外参数
    backtest_kwargs : 传给 VectorEngine 的回测参数（top_n, rebalance_freq, delay 等）
    """

    def __init__(
        self,
        factor_fn:       Callable[..., pd.DataFrame],
        close:           pd.DataFrame,
        is_suspended:    pd.DataFrame,
        is_limit:        pd.DataFrame,
        train_window:    int = 252,
        test_window:     int = 63,
        step:            Optional[int] = None,
        factor_kwargs:   Optional[Dict] = None,
        backtest_kwargs: Optional[Dict] = None,
    ):
        if train_window < 20 or test_window < 5:
            raise ValueError("train_window/test_window 太小，建议 ≥ 20/5")

        self.factor_fn       = factor_fn
        self.close           = close
        self.is_suspended    = is_suspended
        self.is_limit        = is_limit
        self.train_window    = int(train_window)
        self.test_window     = int(test_window)
        self.step            = int(step) if step else int(test_window)
        self.factor_kwargs   = factor_kwargs or {}
        self.backtest_kwargs = backtest_kwargs or {}

    # ------------------------------------------------------------------

    def _make_windows(self, n_dates: int) -> List[tuple]:
        """生成 (train_start, train_end, test_start, test_end) 四元组列表（行索引）。"""
        windows = []
        t = self.train_window
        while t + self.test_window <= n_dates:
            train_start = t - self.train_window
            train_end   = t                     # exclusive
            test_start  = t
            test_end    = min(t + self.test_window, n_dates)
            windows.append((train_start, train_end, test_start, test_end))
            t += self.step
        return windows

    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> WalkForwardReport:
        dates = self.close.index
        n     = len(dates)
        windows = self._make_windows(n)
        if not windows:
            raise ValueError(
                f"数据太短：n={n} < train_window+test_window="
                f"{self.train_window + self.test_window}"
            )

        fold_metrics: List[dict] = []
        fold_results: List[BacktestResult] = []
        fold_returns: List[pd.Series]      = []
        window_dates: List[tuple]          = []

        for k, (ts, te, vs, ve) in enumerate(windows, 1):
            test_start_d = dates[vs]
            test_end_d   = dates[ve - 1]
            train_end_d  = dates[te - 1]
            window_dates.append(
                (dates[ts], train_end_d, test_start_d, test_end_d)
            )

            # 1) 仅用 [0, test_end) 数据算因子（防止用到 OOS 之后的数据）
            close_slice = self.close.iloc[: ve]
            factor = self.factor_fn(close_slice, **self.factor_kwargs)

            # 2) 对 OOS 段单独跑回测
            bt_kwargs = dict(self.backtest_kwargs)
            bt_kwargs["start_date"] = str(test_start_d.date())
            bt_kwargs["end_date"]   = str(test_end_d.date())

            engine = VectorEngine(
                factor       = factor,
                close        = self.close,
                is_suspended = self.is_suspended,
                is_limit     = self.is_limit,
                **bt_kwargs,
            )
            res = engine.run()

            row = {
                "fold":         k,
                "train_end":    train_end_d.date(),
                "test_start":   test_start_d.date(),
                "test_end":     test_end_d.date(),
                "Sharpe_Ratio": res.metrics.get("Sharpe_Ratio", np.nan),
                "年化收益率":    res.metrics.get("年化收益率", np.nan),
                "最大回撤":      res.metrics.get("最大回撤", np.nan),
                "IC_Mean":      res.metrics.get("IC_Mean", np.nan),
                "ICIR":         res.metrics.get("ICIR", np.nan),
                "日均换手率":    res.metrics.get("日均换手率", np.nan),
            }
            fold_metrics.append(row)
            fold_results.append(res)
            fold_returns.append(res.daily_returns)

            if verbose:
                print(f"[WalkForward] fold {k}/{len(windows)}  "
                      f"OOS [{test_start_d.date()} ~ {test_end_d.date()}]  "
                      f"Sharpe={row['Sharpe_Ratio']:+.2f}  "
                      f"IC={row['IC_Mean']:+.4f}")

        summary = pd.DataFrame(fold_metrics).set_index("fold")
        # 拼接全 OOS 净值（去掉重叠日，后窗覆盖前窗）
        all_ret = pd.concat(fold_returns).sort_index()
        all_ret = all_ret[~all_ret.index.duplicated(keep="last")]
        equity  = (1 + all_ret.fillna(0.0)).cumprod()
        equity.name = "oos_equity"

        sharpe_series = summary["Sharpe_Ratio"].dropna()
        return WalkForwardReport(
            windows         = window_dates,
            fold_metrics    = fold_metrics,
            summary         = summary,
            equity_curve    = equity,
            oos_sharpe_mean = float(sharpe_series.mean()) if len(sharpe_series) else float("nan"),
            oos_sharpe_std  = float(sharpe_series.std(ddof=1)) if len(sharpe_series) > 1 else 0.0,
            oos_ic_mean     = float(summary["IC_Mean"].dropna().mean()) if "IC_Mean" in summary else float("nan"),
            fold_results    = fold_results,
        )
