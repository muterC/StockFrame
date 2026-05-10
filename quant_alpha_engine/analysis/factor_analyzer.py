"""
FactorAnalyzer — 多因子诊断工具
================================
在做多因子组合（E1/E2 加权、Fusion 等）之前，必须先看因子之间的相关性，
否则两个 corr=0.95 的因子相加=没有新信息，徒增过拟合风险。

提供：
  - correlation_matrix : 截面 Rank IC 相关矩阵（推荐）或值相关矩阵
  - turnover_consistency: 不同因子之间日 Top-N 持仓的 Jaccard 重合度
  - find_redundant     : 给定阈值，列出高度相关因子对
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Literal, Optional

import numpy as np
import pandas as pd


class FactorAnalyzer:
    """
    多因子相关性 / 一致性诊断（全静态方法）。
    """

    # ------------------------------------------------------------------
    # 相关性矩阵
    # ------------------------------------------------------------------

    @staticmethod
    def correlation_matrix(
        alphas: Dict[str, pd.DataFrame],
        method: Literal["rank", "pearson"] = "rank",
    ) -> pd.DataFrame:
        """
        计算因子之间的截面相关性矩阵（按日均值）。

        - method='rank'   ：每日截面 Spearman 相关（推荐，与 IC 同尺度）
        - method='pearson'：每日截面 Pearson 相关

        Parameters
        ----------
        alphas : {factor_name: T×N DataFrame}
        method : 'rank' 或 'pearson'

        Returns
        -------
        pd.DataFrame : K×K 相关矩阵，对称，对角=1
        """
        if not alphas:
            raise ValueError("alphas 字典为空")
        names = list(alphas.keys())
        K     = len(names)

        # 对齐所有因子到共同 index/columns
        common_idx = None
        common_col = None
        for f in alphas.values():
            common_idx = f.index if common_idx is None else common_idx.intersection(f.index)
            common_col = f.columns if common_col is None else common_col.intersection(f.columns)

        aligned = {n: alphas[n].loc[common_idx, common_col] for n in names}

        # 若需 rank，先做截面 pct rank
        if method == "rank":
            aligned = {n: df.rank(axis=1, pct=True) for n, df in aligned.items()}

        out = pd.DataFrame(np.eye(K), index=names, columns=names)
        for i in range(K):
            for j in range(i + 1, K):
                a = aligned[names[i]]
                b = aligned[names[j]]
                # 每日截面 Pearson 相关，再取时序均值
                a_dm = a.sub(a.mean(axis=1), axis=0)
                b_dm = b.sub(b.mean(axis=1), axis=0)
                num  = (a_dm * b_dm).sum(axis=1)
                den  = np.sqrt((a_dm ** 2).sum(axis=1)) * np.sqrt((b_dm ** 2).sum(axis=1))
                daily_corr = (num / den.replace(0, np.nan)).dropna()
                rho = float(daily_corr.mean()) if len(daily_corr) > 0 else np.nan
                out.iloc[i, j] = rho
                out.iloc[j, i] = rho
        return out

    # ------------------------------------------------------------------
    # 持仓一致性（Top-N 重合度）
    # ------------------------------------------------------------------

    @staticmethod
    def turnover_consistency(
        alphas: Dict[str, pd.DataFrame],
        top_n: int = 30,
    ) -> pd.DataFrame:
        """
        因子之间 Top-N 持仓的 Jaccard 平均重合度。

        每天对每个因子取 Top-N 股票集合，两两计算 |A∩B| / |A∪B|，
        再对所有交易日取均值。值域 [0, 1]，越接近 1 越冗余。

        Returns
        -------
        pd.DataFrame : K×K，对角=1
        """
        names = list(alphas.keys())
        K = len(names)

        # 对齐
        common_idx = None
        common_col = None
        for f in alphas.values():
            common_idx = f.index if common_idx is None else common_idx.intersection(f.index)
            common_col = f.columns if common_col is None else common_col.intersection(f.columns)
        aligned = {n: alphas[n].loc[common_idx, common_col] for n in names}

        # 每天每个因子的 Top-N mask（True = 入选）
        masks: Dict[str, pd.DataFrame] = {}
        for n, f in aligned.items():
            ranks = f.rank(axis=1, ascending=False, method="first")
            masks[n] = (ranks <= top_n)

        out = pd.DataFrame(np.eye(K), index=names, columns=names)
        for i in range(K):
            for j in range(i + 1, K):
                a = masks[names[i]].values
                b = masks[names[j]].values
                inter = (a & b).sum(axis=1)
                union = (a | b).sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    jac = np.where(union > 0, inter / union, np.nan)
                v = float(np.nanmean(jac)) if np.any(~np.isnan(jac)) else np.nan
                out.iloc[i, j] = v
                out.iloc[j, i] = v
        return out

    # ------------------------------------------------------------------
    # 冗余因子检测
    # ------------------------------------------------------------------

    @staticmethod
    def find_redundant(
        corr_matrix: pd.DataFrame,
        threshold: float = 0.8,
    ) -> List[Tuple[str, str, float]]:
        """
        列出相关性高于阈值的因子对（按 |corr| 降序）。

        Returns
        -------
        list of (name_a, name_b, corr)
        """
        pairs = []
        names = list(corr_matrix.index)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                v = corr_matrix.iloc[i, j]
                if pd.notna(v) and abs(v) >= threshold:
                    pairs.append((names[i], names[j], float(v)))
        pairs.sort(key=lambda x: -abs(x[2]))
        return pairs
