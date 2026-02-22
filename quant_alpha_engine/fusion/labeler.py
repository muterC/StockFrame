"""
fusion/labeler.py
=================
因子融合模块 — 标签生成器（Labeler）

将价格矩阵或自定义 DataFrame 转换为监督学习所需的前向收益标签（Y 矩阵）。

Quick Start::

    from quant_alpha_engine.fusion import Labeler

    # 方式一：使用内置收益率标签（最常用）
    labeler = Labeler()
    y = labeler.set_label(
        target='close',
        horizon=5,
        method='return',
        data={'close': close_df, 'volume': volume_df},
    )

    # 方式二：自定义标签（优先级最高）
    my_label = (close.shift(-3) / close - 1)    # 自己构造的 3 日收益率
    y = labeler.set_label(custom_label=my_label)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional


class Labeler:
    """
    前向收益标签生成器。

    两种使用模式：

    1. **内置预设模式**：从价格矩阵自动生成 ``horizon`` 日前向收益/对数收益标签。
       通过 ``data`` 参数传入含价格矩阵的字典，``target`` 指定使用哪个价格列。

    2. **自定义模式**：直接传入已构造好的 ``custom_label`` DataFrame（T×N），
       Labeler 仅做合法性校验后原样返回。**此模式优先级高于内置模式**。

    Attributes
    ----------
    label_ : pd.DataFrame or None
        上一次调用 ``set_label`` 的返回值，方便复用。

    Examples
    --------
    >>> lb = Labeler()
    >>> y = lb.set_label(target='close', horizon=5, method='return',
    ...                  data={'close': close_df})
    >>> y.shape == close_df.shape
    True
    """

    VALID_TARGETS = frozenset(['close', 'open', 'vwap'])
    VALID_METHODS = frozenset(['return', 'log_return'])

    def __init__(self) -> None:
        self.label_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def set_label(
        self,
        target: str = 'close',
        horizon: int = 1,
        method: str = 'return',
        data: Optional[Dict[str, pd.DataFrame]] = None,
        custom_label: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        生成并返回标签矩阵（T×N）。

        Parameters
        ----------
        target : str, default 'close'
            内置预设模式的价格来源：

            * ``'close'``  — 收盘价（最常用）
            * ``'open'``   — 次日开盘价，模拟 T+1 开盘买入
            * ``'vwap'``   — 成交量加权均价（需在 ``data`` 中提供 ``'vwap'`` 键）

        horizon : int, default 1
            前向天数。``horizon=5`` 表示 5 日后的收益率。
            标签在时间轴上 **向前偏移**（shift(-horizon)），
            因此标签矩阵的最后 ``horizon`` 行为 NaN。

        method : str, default 'return'
            收益率计算方式：

            * ``'return'``     — 简单收益率：price.shift(-horizon) / price - 1
            * ``'log_return'`` — 对数收益率：log(price.shift(-horizon) / price)

        data : dict of {str: pd.DataFrame}, optional
            内置预设模式所需的价格矩阵字典。键名须包含 ``target``。
            示例：``{'close': close_df, 'vwap': vwap_df}``

        custom_label : pd.DataFrame, optional
            用户自定义标签矩阵（T×N）。
            **当此参数不为 None 时，以上所有参数均被忽略**，
            Labeler 仅做基础合法性校验后直接返回该矩阵。

        Returns
        -------
        pd.DataFrame
            形状为 (T, N) 的标签矩阵，索引和列名与输入价格矩阵一致。
            最后 ``horizon`` 行由于前向偏移而为 NaN。

        Raises
        ------
        ValueError
            * ``custom_label`` 不是 pd.DataFrame
            * ``target`` 不在 VALID_TARGETS 中
            * ``method`` 不在 VALID_METHODS 中
            * ``horizon`` ≤ 0
            * ``data`` 中缺少 ``target`` 对应的键

        Examples
        --------
        >>> lb = Labeler()
        >>> # 内置模式：5 日简单收益率
        >>> y = lb.set_label(target='close', horizon=5, data={'close': close_df})
        >>> # 内置模式：1 日对数收益率
        >>> y = lb.set_label(method='log_return', data={'close': close_df})
        >>> # 自定义模式：3 日收益率（手工构造）
        >>> y = lb.set_label(custom_label=close_df.pct_change(3).shift(-3))
        """
        # ── 优先：自定义标签 ──────────────────────────────────────────────
        if custom_label is not None:
            self.label_ = self._validate_custom(custom_label)
            return self.label_

        # ── 内置预设模式 ─────────────────────────────────────────────────
        self._validate_params(target, horizon, method, data)
        price: pd.DataFrame = data[target].copy()   # type: ignore[index]
        self.label_ = self._compute_label(price, horizon, method)
        return self.label_

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_label(
        price: pd.DataFrame,
        horizon: int,
        method: str,
    ) -> pd.DataFrame:
        """
        根据价格矩阵计算前向收益率标签。

        Parameters
        ----------
        price   : 价格矩阵 (T × N)，不含 NaN 为最佳
        horizon : 前向天数，>= 1
        method  : 'return' | 'log_return'

        Returns
        -------
        pd.DataFrame
            前向收益率矩阵，最后 ``horizon`` 行为 NaN。
        """
        future_price = price.shift(-horizon)
        safe_price = price.replace(0, np.nan)

        if method == 'return':
            label = future_price / safe_price - 1.0
        else:  # log_return
            ratio = future_price / safe_price
            # 取对数时过滤非正值
            ratio[ratio <= 0] = np.nan
            label = np.log(ratio)

        return label

    @staticmethod
    def _validate_custom(custom_label: object) -> pd.DataFrame:
        """
        校验用户自定义标签格式。

        Parameters
        ----------
        custom_label : 任意对象，期望为 pd.DataFrame

        Returns
        -------
        pd.DataFrame
            原样返回（副本），确保列名和索引类型不变。

        Raises
        ------
        ValueError
            若 ``custom_label`` 不是 pd.DataFrame。
        """
        if not isinstance(custom_label, pd.DataFrame):
            raise ValueError(
                f"custom_label 必须是 pd.DataFrame，"
                f"当前类型: {type(custom_label).__name__}"
            )
        if custom_label.empty:
            raise ValueError("custom_label 不能为空 DataFrame。")
        return custom_label.copy()

    @classmethod
    def _validate_params(
        cls,
        target: str,
        horizon: int,
        method: str,
        data: Optional[Dict[str, pd.DataFrame]],
    ) -> None:
        """
        校验内置预设模式的参数合法性。

        Raises
        ------
        ValueError
            任一参数不合法时抛出，错误信息指明具体原因。
        """
        if target not in cls.VALID_TARGETS:
            raise ValueError(
                f"target='{target}' 不合法，"
                f"可选值: {sorted(cls.VALID_TARGETS)}"
            )
        if method not in cls.VALID_METHODS:
            raise ValueError(
                f"method='{method}' 不合法，"
                f"可选值: {sorted(cls.VALID_METHODS)}"
            )
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(
                f"horizon 必须是正整数，当前值: {horizon!r}"
            )
        if data is None:
            raise ValueError(
                "内置预设模式需要通过 data 参数传入价格矩阵字典，"
                "例如 data={'close': close_df}。"
                "若使用自定义标签，请改用 custom_label 参数。"
            )
        if not isinstance(data, dict):
            raise ValueError(
                f"data 必须是 dict 类型，当前类型: {type(data).__name__}"
            )
        if target not in data:
            raise ValueError(
                f"data 字典中缺少 key='{target}'，"
                f"当前 keys: {list(data.keys())}"
            )
        price = data[target]
        if not isinstance(price, pd.DataFrame):
            raise ValueError(
                f"data['{target}'] 必须是 pd.DataFrame，"
                f"当前类型: {type(price).__name__}"
            )
        if price.empty:
            raise ValueError(f"data['{target}'] 不能为空 DataFrame。")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def from_price(
        price: pd.DataFrame,
        horizon: int = 1,
        method: str = 'return',
    ) -> pd.DataFrame:
        """
        便捷静态方法：直接从价格矩阵生成标签，无需实例化 Labeler。

        Parameters
        ----------
        price   : 价格矩阵 (T × N)
        horizon : 前向天数，默认 1
        method  : 'return' | 'log_return'，默认 'return'

        Returns
        -------
        pd.DataFrame
            前向收益率标签矩阵。

        Examples
        --------
        >>> y = Labeler.from_price(close_df, horizon=5)
        """
        if not isinstance(price, pd.DataFrame) or price.empty:
            raise ValueError("price 必须是非空 pd.DataFrame。")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"horizon 必须是正整数，当前值: {horizon!r}")
        if method not in Labeler.VALID_METHODS:
            raise ValueError(
                f"method='{method}' 不合法，可选值: {sorted(Labeler.VALID_METHODS)}"
            )
        return Labeler._compute_label(price, horizon, method)

    def __repr__(self) -> str:
        label_info = (
            f"shape={self.label_.shape}"
            if self.label_ is not None
            else "未生成"
        )
        return f"Labeler(label_={label_info})"
