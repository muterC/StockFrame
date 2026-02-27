"""
fusion/combiner.py
==================
因子融合框架 — FactorCombiner 抽象基类 + 两个具体实现

模块结构
--------
::

    FactorCombiner (ABC)           ← 抽象基类：对齐、评估、持久化
    ├── StatisticalCombiner        ← 统计融合：等权 / IC加权 / 最小方差
    └── MLCombiner                 ← 机器学习融合：Expanding Window，防未来函数

Quick Start::

    from quant_alpha_engine.fusion import StatisticalCombiner, MLCombiner

    # ── 统计融合 ─────────────────────────────────────────────────────
    stat = StatisticalCombiner(method='ic_weighted')
    stat.fit([f1, f2, f3], label=y)
    print(stat.weights_)            # IC 加权权重向量
    result = stat.evaluate(
        factors=[f1, f2, f3],
        close=close, is_suspended=susp, is_limit=limit,
        rebalance_freq=5, top_n=30,
    )
    result.print_summary()

    # ── ML 融合 ───────────────────────────────────────────────────────
    ml = MLCombiner(model_type='ridge', min_train_periods=60, refit_freq=20)
    ml.fit([f1, f2, f3], label=y)
    pred = ml.predict([f1, f2, f3])
    pred.iloc[:60].isna().all().all()   # True，前 60 行无预测（积累期）
    print(ml.feature_importances_)
    ml.save("models/my_ml_combiner.pkl")
    loaded = MLCombiner.load("models/my_ml_combiner.pkl")

Notes
-----
- 所有 combiner 对因子先做 ``rank(axis=1, pct=True)`` 截面归一化，消除量纲差异
- MLCombiner 严格使用 Expanding Window，任何截面仅用历史数据训练，不产生未来函数
- XGBoost 为可选依赖（try/except 降级）；若未安装，指定 'xgboost' 时报明确错误
"""

from __future__ import annotations

import os
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# sklearn 必选
try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# xgboost 可选
try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from quant_alpha_engine.backtest.vector_engine import VectorEngine, BacktestResult


# ===========================================================================
# 类型别名
# ===========================================================================

FactorList = List[pd.DataFrame]
StatMethod = Literal["equal", "ic_weighted", "min_variance"]
MLModel    = Literal["linear", "ridge", "random_forest", "xgboost"]


# ===========================================================================
# 抽象基类
# ===========================================================================

class FactorCombiner(ABC):
    """
    因子融合器抽象基类。

    所有子类须实现 ``fit`` 和 ``predict`` 两个抽象方法。
    基类提供数据对齐、回测评估和模型持久化等公共功能。

    Attributes
    ----------
    _is_fitted : bool
        是否已调用过 ``fit``。
    _factor_names : list of str or None
        训练时因子的名称列表（长度 = K，因子数量）。
    """

    def __init__(self) -> None:
        self._is_fitted: bool = False
        self._factor_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # 抽象接口（子类必须实现）
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        factors: FactorList,
        label: pd.DataFrame,
    ) -> "FactorCombiner":
        """
        在历史数据上拟合模型。

        Parameters
        ----------
        factors : list of pd.DataFrame, each shape (T, N)
            K 个因子矩阵列表，每个矩阵形状相同（T×N）。
        label : pd.DataFrame, shape (T, N)
            前向收益率标签矩阵（由 Labeler 生成或手工构造）。

        Returns
        -------
        self
            支持链式调用：``combiner.fit(...).predict(...)``。
        """

    @abstractmethod
    def predict(
        self,
        factors: FactorList,
    ) -> pd.DataFrame:
        """
        生成合成因子矩阵。

        Parameters
        ----------
        factors : list of pd.DataFrame, each shape (T, N)
            与 ``fit`` 时相同格式的因子矩阵列表。

        Returns
        -------
        pd.DataFrame, shape (T, N)
            合成因子矩阵，可直接传入 ``VectorEngine``。
        """

    # ------------------------------------------------------------------
    # 公共方法：评估（调用 VectorEngine）
    # ------------------------------------------------------------------

    def evaluate(
        self,
        factors: FactorList,
        close: pd.DataFrame,
        is_suspended: pd.DataFrame,
        is_limit: pd.DataFrame,
        rebalance_freq: int = 1,
        top_n: int = 50,
        weight_method: str = "equal",
        cost_rate: float = 0.0015,
        initial_capital: float = 1_000_000.0,
        delay: int = 0,
        decay: int = 0,
        industry=None,
        start_date=None,
        end_date=None,
    ) -> BacktestResult:
        """
        将合成因子传入 VectorEngine 进行全量回测，返回 BacktestResult。

        先调用 ``predict`` 生成合成因子，再构造 ``VectorEngine`` 运行回测。
        返回结果与单因子回测完全兼容，可直接调用 ``.print_summary()`` 和 ``.plot()``。

        Parameters
        ----------
        factors        : 因子列表，将先经过 ``predict`` 融合为单一因子
        close          : 收盘价矩阵 (T × N)
        is_suspended   : 停牌状态矩阵 (T × N, bool)
        is_limit       : 涨跌停状态矩阵 (T × N, bool)
        rebalance_freq : 调仓频率（天数），默认 1
        top_n          : 持仓股票数量，默认 50
        weight_method  : 'equal' 或 'factor_weighted'
        cost_rate      : 单边交易成本，默认 0.0015
        initial_capital: 初始资金（仅展示用），默认 1,000,000
        delay          : 因子延迟天数，默认 0（融合器输出通常已处理好时序，无需再延迟）
        decay          : 线性衰减窗口，默认 0（不衰减）
        industry       : 行业映射，默认 None（不做中性化）
        start_date     : 回测开始日期（str，如 '2022-06-01'），None=不限制（v2.1 新增）
        end_date       : 回测结束日期（str，如 '2023-12-31'），None=不限制（v2.1 新增）

        Returns
        -------
        BacktestResult
            标准回测结果，与单因子 VectorEngine 输出完全一致。

        Raises
        ------
        RuntimeError
            若 ``fit`` 尚未调用（``_is_fitted=False``）。
        """
        self._check_fitted()
        synthetic_factor = self.predict(factors)
        engine = VectorEngine(
            factor=synthetic_factor,
            close=close,
            is_suspended=is_suspended,
            is_limit=is_limit,
            rebalance_freq=rebalance_freq,
            top_n=top_n,
            weight_method=weight_method,
            cost_rate=cost_rate,
            initial_capital=initial_capital,
            delay=delay,
            decay=decay,
            industry=industry,
            start_date=start_date,
            end_date=end_date,
        )
        return engine.run()

    # ------------------------------------------------------------------
    # 公共方法：持久化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        将 combiner 对象序列化到文件（pickle，protocol 4）。

        自动创建目录（若不存在）。支持 ``.pkl`` / ``.pickle`` / 任意扩展名。

        Parameters
        ----------
        path : str
            保存路径，例如 ``"models/stat_combiner.pkl"``。

        Examples
        --------
        >>> combiner.save("outputs/my_combiner.pkl")
        """
        dir_name = os.path.dirname(os.path.abspath(path))
        os.makedirs(dir_name, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)

    @classmethod
    def load(cls, path: str) -> "FactorCombiner":
        """
        从文件加载 combiner 对象。

        Parameters
        ----------
        path : str
            保存路径，与 ``save`` 时传入的路径一致。

        Returns
        -------
        FactorCombiner
            恢复的 combiner 实例（具体子类类型）。

        Raises
        ------
        FileNotFoundError
            若文件不存在。
        TypeError
            若反序列化后对象不是 FactorCombiner 实例。

        Examples
        --------
        >>> loaded = MLCombiner.load("outputs/my_combiner.pkl")
        >>> loaded._is_fitted
        True
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到文件: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, FactorCombiner):
            raise TypeError(
                f"反序列化对象类型 {type(obj).__name__} 不是 FactorCombiner 子类"
            )
        return obj

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _align_factors_label(
        factors: FactorList,
        label: pd.DataFrame,
    ) -> Tuple[FactorList, pd.DataFrame]:
        """
        取因子列表和标签矩阵的 index / columns 交集，对齐数据。

        Parameters
        ----------
        factors : list of pd.DataFrame (K 个，T×N)
        label   : pd.DataFrame (T×N)

        Returns
        -------
        (aligned_factors, aligned_label)
            对齐后的因子列表和标签矩阵。
        """
        common_idx = label.index
        common_cols = label.columns
        for f in factors:
            common_idx  = common_idx.intersection(f.index)
            common_cols = common_cols.intersection(f.columns)
        aligned_label   = label.loc[common_idx, common_cols]
        aligned_factors = [f.loc[common_idx, common_cols] for f in factors]
        return aligned_factors, aligned_label

    @staticmethod
    def _align_factors_only(
        factors: FactorList,
    ) -> FactorList:
        """
        取因子列表的 index / columns 交集，对齐各因子数据。

        Parameters
        ----------
        factors : list of pd.DataFrame (K 个，T×N)

        Returns
        -------
        list of pd.DataFrame
            对齐后的因子列表。
        """
        if not factors:
            return []
        common_idx  = factors[0].index
        common_cols = factors[0].columns
        for f in factors[1:]:
            common_idx  = common_idx.intersection(f.index)
            common_cols = common_cols.intersection(f.columns)
        return [f.loc[common_idx, common_cols] for f in factors]

    @staticmethod
    def _rank_normalize(factors: FactorList) -> FactorList:
        """
        对每个因子做截面百分位排名，消除量纲差异。

        Parameters
        ----------
        factors : list of pd.DataFrame (K 个，T×N)

        Returns
        -------
        list of pd.DataFrame
            每个值域为 (0, 1] 的排名矩阵。
        """
        return [f.rank(axis=1, pct=True, na_option='keep') for f in factors]

    def _check_fitted(self) -> None:
        """断言 fit 已被调用，否则抛出 RuntimeError。"""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} 尚未 fit，"
                "请先调用 .fit(factors, label) 再使用 .predict() / .evaluate()。"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"is_fitted={self._is_fitted}, "
            f"n_factors={len(self._factor_names) if self._factor_names else None})"
        )


# ===========================================================================
# 统计融合器
# ===========================================================================

class StatisticalCombiner(FactorCombiner):
    """
    基于统计方法的因子融合器。

    提供三种权重计算方式，均在截面 Rank 归一化后进行融合：

    * ``'equal'``        — 等权融合：每个因子权重相同（1/K）
    * ``'ic_weighted'``  — IC 加权：权重正比于各因子的平均 Rank IC
    * ``'min_variance'`` — 最小方差：SLSQP 求解最小化合成因子方差，权重之和为 1

    Parameters
    ----------
    method : str, default 'equal'
        权重计算方式，可选 'equal' / 'ic_weighted' / 'min_variance'。

    Attributes
    ----------
    weights_ : np.ndarray, shape (K,)
        fit 后的融合权重（归一化，总和为 1）。
    ic_matrix_ : pd.DataFrame, shape (T, K)
        fit 后各因子每日截面 Rank IC。仅 'ic_weighted' 和 'min_variance' 时有效。

    Examples
    --------
    >>> stat = StatisticalCombiner('ic_weighted')
    >>> stat.fit([f1, f2, f3], label=y)
    >>> print(stat.weights_)             # [0.35, 0.40, 0.25]
    >>> result = stat.evaluate([f1, f2, f3], close=close, ...)
    """

    VALID_METHODS: frozenset = frozenset(["equal", "ic_weighted", "min_variance"])

    def __init__(self, method: StatMethod = "equal") -> None:
        super().__init__()
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method='{method}' 不合法，可选值: {sorted(self.VALID_METHODS)}"
            )
        self.method = method
        self.weights_: Optional[np.ndarray] = None
        self.ic_matrix_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 抽象方法实现
    # ------------------------------------------------------------------

    def fit(
        self,
        factors: FactorList,
        label: pd.DataFrame,
    ) -> "StatisticalCombiner":
        """
        计算融合权重。

        Parameters
        ----------
        factors : list of pd.DataFrame, each (T, N)
        label   : pd.DataFrame (T, N)，前向收益率标签

        Returns
        -------
        self
        """
        if not factors:
            raise ValueError("factors 列表不能为空。")
        factors_aligned, label_aligned = self._align_factors_label(factors, label)
        factors_ranked = self._rank_normalize(factors_aligned)
        K = len(factors_ranked)
        self._factor_names = [f"factor_{i}" for i in range(K)]

        if self.method == "equal":
            self.weights_ = np.ones(K) / K

        elif self.method in ("ic_weighted", "min_variance"):
            # 计算每日截面 Rank IC：corr(因子截面排名, 标签截面排名)
            ic_cols = {}
            label_ranked = label_aligned.rank(axis=1, pct=True, na_option='keep')
            for i, f in enumerate(factors_ranked):
                daily_ic = f.corrwith(label_ranked, axis=1)  # Pearson ≈ Rank IC
                ic_cols[self._factor_names[i]] = daily_ic
            self.ic_matrix_ = pd.DataFrame(ic_cols).dropna(how='all')

            if self.method == "ic_weighted":
                ic_means = self.ic_matrix_.mean()
                sum_abs  = ic_means.abs().sum()
                if sum_abs < 1e-10:
                    # 所有 IC 接近零，退化为等权
                    warnings.warn(
                        "所有因子平均 IC 接近零，StatisticalCombiner 退化为等权融合。",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self.weights_ = np.ones(K) / K
                else:
                    raw_w = ic_means.values / sum_abs
                    # 确保权重全正（负 IC 取绝对值方向融合，此处取原始方向）
                    self.weights_ = raw_w / np.abs(raw_w).sum()

            else:  # min_variance
                self.weights_ = self._solve_min_variance(
                    factors_ranked, K
                )

        self._is_fitted = True
        return self

    def predict(self, factors: FactorList) -> pd.DataFrame:
        """
        生成合成因子矩阵（加权平均排名归一化因子）。

        Parameters
        ----------
        factors : list of pd.DataFrame, each (T, N)

        Returns
        -------
        pd.DataFrame (T, N)
        """
        self._check_fitted()
        factors_aligned = self._align_factors_only(factors)
        factors_ranked  = self._rank_normalize(factors_aligned)
        K = len(factors_ranked)
        if K != len(self.weights_):  # type: ignore[arg-type]
            raise ValueError(
                f"predict 时因子数量 ({K}) 与 fit 时 ({len(self.weights_)}) 不一致。"
            )
        ref = factors_ranked[0]
        result = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)
        for w, f in zip(self.weights_, factors_ranked):  # type: ignore[arg-type]
            result = result.add(f.fillna(0.0) * w, fill_value=0.0)
        # 将所有因子均为 NaN 的格子还原为 NaN
        any_valid = pd.concat(
            [f.notna() for f in factors_ranked], axis=0
        ).groupby(level=0).any()
        result[~any_valid] = np.nan
        return result

    # ------------------------------------------------------------------
    # 内部：最小方差求解
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_min_variance(
        factors_ranked: FactorList,
        K: int,
    ) -> np.ndarray:
        """
        SLSQP 求解最小方差权重：min w^T·Cov·w，s.t. Σwᵢ=1, wᵢ≥0。

        将因子展平为 (T×N, K) 样本矩阵，计算协方差，再求解 QP 问题。

        Parameters
        ----------
        factors_ranked : K 个已 Rank 归一化的因子矩阵
        K              : 因子数量

        Returns
        -------
        np.ndarray, shape (K,)
            最优权重向量。
        """
        # 展平为二维矩阵 (T*N, K)，过滤含 NaN 行
        flat_list = [f.values.ravel() for f in factors_ranked]
        data_2d = np.column_stack(flat_list)       # (T*N, K)
        valid_rows = ~np.isnan(data_2d).any(axis=1)
        data_clean = data_2d[valid_rows]            # (M, K)

        if data_clean.shape[0] < K * 2:
            warnings.warn(
                "有效样本数不足，min_variance 退化为等权融合。",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.ones(K) / K

        cov = np.cov(data_clean, rowvar=False)    # (K, K)
        w0  = np.ones(K) / K

        def objective(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        def grad(w: np.ndarray) -> np.ndarray:
            return 2.0 * cov @ w

        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        bounds = [(0.0, 1.0)] * K

        res = minimize(
            objective,
            w0,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if res.success:
            w = np.maximum(res.x, 0.0)
            w /= w.sum()
            return w
        else:
            warnings.warn(
                f"最小方差优化未收敛（{res.message}），退化为等权融合。",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.ones(K) / K

    def __repr__(self) -> str:
        w_str = (
            f"weights_={np.round(self.weights_, 4).tolist()}"
            if self.weights_ is not None
            else "weights_=None"
        )
        return (
            f"StatisticalCombiner(method='{self.method}', "
            f"is_fitted={self._is_fitted}, {w_str})"
        )


# ===========================================================================
# 机器学习融合器
# ===========================================================================

class MLCombiner(FactorCombiner):
    """
    基于机器学习的因子融合器（Expanding Window，严格防未来函数）。

    使用 sklearn 接口的模型，在历史数据上做 Expanding Window 滚动训练：

    * 前 ``min_train_periods`` 天无预测（积累期，输出 NaN）
    * 此后每隔 ``refit_freq`` 天重新训练一次模型
    * 训练集持续扩大（expanding），测试集为未来 ``refit_freq`` 天

    支持的模型类型：

    * ``'linear'``        — 普通线性回归（sklearn LinearRegression）
    * ``'ridge'``         — 岭回归（sklearn Ridge），默认 alpha=1.0
    * ``'random_forest'`` — 随机森林（sklearn RandomForestRegressor）
    * ``'xgboost'``       — XGBoost（需单独安装 xgboost 包）

    Parameters
    ----------
    model_type         : 模型类型
    min_train_periods  : 最小训练样本数（天数），默认 60
    refit_freq         : 重新训练频率（天数），默认 20
    ridge_alpha        : Ridge 正则化系数，默认 1.0
    rf_n_estimators    : 随机森林树数量，默认 100
    rf_max_depth       : 随机森林最大深度，默认 5
    xgb_n_estimators   : XGBoost 树数量，默认 100
    xgb_max_depth      : XGBoost 最大深度，默认 3
    xgb_learning_rate  : XGBoost 学习率，默认 0.1

    Attributes
    ----------
    feature_importances_ : pd.Series or None
        fit 后各因子的重要性（归一化到 [0, 1]），index 为因子名称。
        - Linear/Ridge → ``|coef_|`` 归一化
        - RF/XGBoost   → ``feature_importances_`` 归一化
    _models : list of fitted estimator
        各训练窗口的模型列表（供序列化和调试）。

    Examples
    --------
    >>> ml = MLCombiner('ridge', min_train_periods=60, refit_freq=20)
    >>> ml.fit([f1, f2, f3], label=y)
    >>> pred = ml.predict([f1, f2, f3])
    >>> pred.iloc[:60].isna().all().all()   # True
    >>> print(ml.feature_importances_)
    >>> ml.save("models/ml_combiner.pkl")
    """

    def __init__(
        self,
        model_type: MLModel = "ridge",
        min_train_periods: int = 60,
        refit_freq: int = 20,
        ridge_alpha: float = 1.0,
        rf_n_estimators: int = 100,
        rf_max_depth: int = 5,
        xgb_n_estimators: int = 100,
        xgb_max_depth: int = 3,
        xgb_learning_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self._check_deps(model_type)
        self.model_type         = model_type
        self.min_train_periods  = min_train_periods
        self.refit_freq         = refit_freq
        self.ridge_alpha        = ridge_alpha
        self.rf_n_estimators    = rf_n_estimators
        self.rf_max_depth       = rf_max_depth
        self.xgb_n_estimators   = xgb_n_estimators
        self.xgb_max_depth      = xgb_max_depth
        self.xgb_learning_rate  = xgb_learning_rate

        self.feature_importances_: Optional[pd.Series] = None
        self._models: List = []
        self._predictions: Optional[pd.DataFrame] = None   # 训练期缓存的预测值

    # ------------------------------------------------------------------
    # 依赖检查
    # ------------------------------------------------------------------

    @staticmethod
    def _check_deps(model_type: str) -> None:
        """检查所需依赖是否已安装。"""
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "MLCombiner 需要 scikit-learn。"
                "请运行: pip install scikit-learn>=1.3.0"
            )
        if model_type == "xgboost" and not _XGB_AVAILABLE:
            raise ImportError(
                "model_type='xgboost' 需要安装 xgboost 包。"
                "请运行: pip install xgboost>=1.7.0"
            )

    # ------------------------------------------------------------------
    # 模型工厂
    # ------------------------------------------------------------------

    def _make_model(self) -> object:
        """根据 model_type 创建新的未训练模型实例。"""
        if self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "ridge":
            return Ridge(alpha=self.ridge_alpha)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                n_jobs=-1,
                random_state=42,
            )
        elif self.model_type == "xgboost":
            return XGBRegressor(
                n_estimators=self.xgb_n_estimators,
                max_depth=self.xgb_max_depth,
                learning_rate=self.xgb_learning_rate,
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            )
        else:
            raise ValueError(f"未知 model_type: {self.model_type}")

    # ------------------------------------------------------------------
    # 特征工程：(T, N, K) → (T*N, K)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_feature_matrix(
        factors_ranked: FactorList,
        t_slice: Optional[slice] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将 K 个因子矩阵展平为 (M, K) 特征矩阵（过滤含 NaN 行）。

        Parameters
        ----------
        factors_ranked : Rank 归一化后的因子列表，每个 shape (T, N)
        t_slice        : 时间轴切片，若为 None 则使用全部数据

        Returns
        -------
        (X_clean, valid_mask, T_size)
            - X_clean    : shape (M, K)，已过滤 NaN 的特征矩阵
            - valid_mask : shape (T*N,)，标记哪些行有效
            - T_size     : 实际使用的 T（行数）
        """
        sliced = [f.values[t_slice] if t_slice else f.values for f in factors_ranked]
        T_size = sliced[0].shape[0]
        flat   = np.column_stack([s.ravel() for s in sliced])  # (T*N, K)
        valid  = ~np.isnan(flat).any(axis=1)
        return flat[valid], valid, T_size

    @staticmethod
    def _build_label_vector(
        label_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将标签矩阵展平为向量，返回有效行掩码。

        Parameters
        ----------
        label_arr : np.ndarray, shape (T, N)

        Returns
        -------
        (y_flat, valid_mask)
        """
        y_flat = label_arr.ravel()
        valid  = ~np.isnan(y_flat)
        return y_flat, valid

    # ------------------------------------------------------------------
    # 抽象方法实现
    # ------------------------------------------------------------------

    def fit(
        self,
        factors: FactorList,
        label: pd.DataFrame,
    ) -> "MLCombiner":
        """
        Expanding Window 训练 ML 模型，缓存预测矩阵。

        训练策略
        --------
        ::

            Day  0 ~ min_train_periods-1   → NaN（积累期）
            Day  min_train_periods ~ end   → 每 refit_freq 天重训一次，
                                             训练集 = [0, train_end)，
                                             预测窗口 = [train_end, train_end + refit_freq)

        Parameters
        ----------
        factors : list of pd.DataFrame (K 个，T×N)
        label   : pd.DataFrame (T×N)，前向收益率标签

        Returns
        -------
        self
        """
        if not factors:
            raise ValueError("factors 列表不能为空。")
        factors_aligned, label_aligned = self._align_factors_label(factors, label)
        factors_ranked = self._rank_normalize(factors_aligned)
        K = len(factors_ranked)
        self._factor_names = [f"factor_{i}" for i in range(K)]

        T  = factors_ranked[0].shape[0]
        N  = factors_ranked[0].shape[1]
        idx  = factors_ranked[0].index
        cols = factors_ranked[0].columns

        # 初始化预测矩阵（全 NaN）
        pred_arr = np.full((T, N), np.nan, dtype=np.float64)

        label_arr = label_aligned.values  # (T, N)

        # Expanding Window 训练循环
        self._models = []
        importances_list: List[np.ndarray] = []

        train_end_dates = list(range(
            self.min_train_periods,
            T + 1,
            self.refit_freq,
        ))
        if not train_end_dates or train_end_dates[-1] < T:
            # 确保覆盖所有数据
            train_end_dates.append(T)

        for train_end in train_end_dates:
            if train_end > T:
                break

            # 训练窗口：[0, train_end)
            train_slice = slice(0, train_end)
            X_train_all, valid_x, _ = self._build_feature_matrix(
                factors_ranked, t_slice=train_slice
            )
            y_train_all, valid_y = self._build_label_vector(
                label_arr[train_slice]
            )
            # 同时满足 X 和 y 均有效
            valid_xy = valid_x & valid_y
            X_train  = (np.column_stack([
                f.values[train_slice].ravel() for f in factors_ranked
            ])[valid_xy])
            y_train  = y_train_all[valid_xy]

            if len(X_train) < 10:
                # 样本太少，跳过
                continue

            model = self._make_model()
            model.fit(X_train, y_train)
            self._models.append(model)

            # 收集特征重要性（最后一次训练）
            importances_list.append(
                self._extract_importance(model, K)
            )

            # 预测窗口：[train_end, train_end + refit_freq)
            pred_start = train_end
            pred_end   = min(train_end + self.refit_freq, T)
            if pred_start >= T:
                break

            pred_slice = slice(pred_start, pred_end)
            T_pred = pred_end - pred_start

            for t_local in range(T_pred):
                t_global = pred_start + t_local
                # 单截面特征向量：shape (N, K)
                x_cross = np.column_stack(
                    [f.values[t_global] for f in factors_ranked]
                )                                    # (N, K)
                valid_n = ~np.isnan(x_cross).any(axis=1)  # (N,)
                if valid_n.sum() == 0:
                    continue
                preds_n = np.full(N, np.nan)
                preds_n[valid_n] = model.predict(x_cross[valid_n])
                pred_arr[t_global] = preds_n

        # 保存特征重要性（使用最后一批次的平均）
        if importances_list:
            avg_imp = np.mean(importances_list, axis=0)
            self.feature_importances_ = pd.Series(
                avg_imp / (avg_imp.sum() + 1e-12),
                index=self._factor_names,
                name="feature_importance",
            )

        self._predictions = pd.DataFrame(pred_arr, index=idx, columns=cols)
        self._is_fitted = True
        return self

    def predict(self, factors: FactorList) -> pd.DataFrame:
        """
        返回合成因子矩阵。

        若 ``factors`` 与 ``fit`` 时的数据完全一致（相同 index/columns），
        直接返回训练期缓存的预测矩阵（最快）；
        否则对新数据做滚动推断（使用最后一个训练好的模型，适用于 walk-forward 场景）。

        Parameters
        ----------
        factors : list of pd.DataFrame (K 个，T×N)

        Returns
        -------
        pd.DataFrame (T, N)
            前 min_train_periods 行为 NaN，之后为模型预测的合成因子值。
        """
        self._check_fitted()
        K = len(factors)
        if K != len(self._factor_names):  # type: ignore[arg-type]
            raise ValueError(
                f"predict 时因子数量 ({K}) 与 fit 时 ({len(self._factor_names)}) 不一致。"
            )

        # ── 快速路径：使用缓存 ───────────────────────────────────────────
        if self._predictions is not None:
            try:
                factors_aligned = self._align_factors_only(factors)
                f0 = factors_aligned[0]
                if (
                    f0.index.equals(self._predictions.index)
                    and f0.columns.equals(self._predictions.columns)
                ):
                    return self._predictions.copy()
            except Exception:
                pass

        # ── 慢速路径：用最后训练的模型对新数据推断 ──────────────────────
        if not self._models:
            raise RuntimeError("模型列表为空，请先调用 fit()。")
        last_model  = self._models[-1]
        factors_aligned = self._align_factors_only(factors)
        factors_ranked  = self._rank_normalize(factors_aligned)
        ref  = factors_ranked[0]
        T, N = ref.shape
        pred_arr = np.full((T, N), np.nan, dtype=np.float64)

        for t in range(self.min_train_periods, T):
            x_cross = np.column_stack([f.values[t] for f in factors_ranked])  # (N, K)
            valid_n = ~np.isnan(x_cross).any(axis=1)
            if valid_n.sum() == 0:
                continue
            preds_n = np.full(N, np.nan)
            preds_n[valid_n] = last_model.predict(x_cross[valid_n])
            pred_arr[t] = preds_n

        return pd.DataFrame(pred_arr, index=ref.index, columns=ref.columns)

    # ------------------------------------------------------------------
    # 特征重要性提取
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_importance(model: object, K: int) -> np.ndarray:
        """
        从训练好的模型中提取原始特征重要性（未归一化）。

        Parameters
        ----------
        model : fitted sklearn / xgboost estimator
        K     : 特征数量（因子数）

        Returns
        -------
        np.ndarray, shape (K,)
        """
        if hasattr(model, "coef_"):
            # Linear / Ridge
            coef = np.asarray(model.coef_).ravel()
            return np.abs(coef[:K])
        elif hasattr(model, "feature_importances_"):
            # RandomForest / XGBoost
            return np.asarray(model.feature_importances_[:K])
        else:
            return np.ones(K) / K

    # ------------------------------------------------------------------
    # 扩展持久化：可选缓存预测矩阵
    # ------------------------------------------------------------------

    def save(  # type: ignore[override]
        self,
        path: str,
        save_predictions: bool = True,
    ) -> None:
        """
        序列化 MLCombiner 对象到文件。

        Parameters
        ----------
        path             : 保存路径
        save_predictions : bool, default True
            - True  → 包含预测缓存矩阵（文件较大，加载后 predict 速度更快）
            - False → 仅保存模型权重（文件更小，加载后首次 predict 需重新推断）
        """
        if not save_predictions:
            # 临时移除预测缓存
            old_preds = self._predictions
            self._predictions = None
            super().save(path)
            self._predictions = old_preds
        else:
            super().save(path)

    def __repr__(self) -> str:
        imp_str = (
            f"feature_importances_={self.feature_importances_.round(4).to_dict()}"
            if self.feature_importances_ is not None
            else "feature_importances_=None"
        )
        return (
            f"MLCombiner(model_type='{self.model_type}', "
            f"min_train_periods={self.min_train_periods}, "
            f"refit_freq={self.refit_freq}, "
            f"is_fitted={self._is_fitted}, "
            f"n_models={len(self._models)}, "
            f"{imp_str})"
        )
