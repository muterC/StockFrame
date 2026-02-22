"""
数据验证模块 - 纯验证层，无状态，无外部存储依赖

职责：
- 基础字段验证（OHLCV、复权因子、市值）
- 重复数据检测
- 新旧数据对比与合并（新数据优先覆盖重叠日期）

设计原则：
- 所有方法均为 @staticmethod，无实例状态
- 不依赖外部存储（DataRepository）
- 可独立使用，也可被 DataRepository / DataUpdater 调用
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class DataValidator:
    """
    纯验证类，所有方法为 @staticmethod，无实例状态，无外部依赖。

    提供：
    - 基础字段验证（validate_ohlcv / validate_adjustment_factors / validate_market_value）
    - 重复索引检测（check_duplicates）
    - 新旧数据对比与合并（compare_and_merge）
    """

    # ==================== 基础验证方法 ====================

    @staticmethod
    def check_duplicates(data: pd.DataFrame) -> pd.DataFrame:
        """
        检测重复索引行。

        Args:
            data: 待检查的 DataFrame（索引应为日期）

        Returns:
            包含所有重复行的 DataFrame；若无重复则返回空 DataFrame。
        """
        return data[data.index.duplicated(keep=False)]

    @staticmethod
    def validate_ohlcv(
        data: pd.DataFrame,
        raise_exception: bool = False
    ) -> tuple[bool, list[str]]:
        """
        验证 OHLCV 数据的完整性和价格逻辑关系。

        Args:
            data: OHLCV DataFrame，索引为日期
            raise_exception: True 时验证失败抛出 RuntimeError

        Returns:
            (is_valid, errors)
                is_valid: 验证是否全部通过
                errors:   错误描述列表（通过时为空列表）

        Raises:
            RuntimeError: raise_exception=True 且存在错误时
        """
        errors: list[str] = []

        # 1. 检查必需列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"缺少必需列: {missing_cols}")
            # 缺列时无法继续后续检查
            if raise_exception:
                raise RuntimeError("; ".join(errors))
            return False, errors

        # 2. 检查数据类型
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"列 {col} 不是数值类型")

        # 3. 检查缺失值
        null_counts = data[required_cols].isnull().sum()
        if null_counts.any():
            errors.append(f"存在空值: {null_counts[null_counts > 0].to_dict()}")

        # 4. 价格逻辑关系
        invalid_high_low = (data['high'] < data['low']).sum()
        if invalid_high_low > 0:
            errors.append(f"发现 {invalid_high_low} 行 high < low")

        invalid_high_open = (data['high'] < data['open']).sum()
        if invalid_high_open > 0:
            errors.append(f"发现 {invalid_high_open} 行 high < open")

        invalid_high_close = (data['high'] < data['close']).sum()
        if invalid_high_close > 0:
            errors.append(f"发现 {invalid_high_close} 行 high < close")

        invalid_low_open = (data['low'] > data['open']).sum()
        if invalid_low_open > 0:
            errors.append(f"发现 {invalid_low_open} 行 low > open")

        invalid_low_close = (data['low'] > data['close']).sum()
        if invalid_low_close > 0:
            errors.append(f"发现 {invalid_low_close} 行 low > close")

        # 5. 负值检查
        negative_prices = (data[['open', 'high', 'low', 'close']] < 0).any(axis=1).sum()
        if negative_prices > 0:
            errors.append(f"发现 {negative_prices} 行价格为负")

        negative_volume = (data['volume'] < 0).sum()
        if negative_volume > 0:
            errors.append(f"发现 {negative_volume} 行成交量为负")

        # 6. 重复索引
        dup_df = DataValidator.check_duplicates(data)
        if not dup_df.empty:
            errors.append(f"发现 {len(dup_df)} 行重复索引")

        if errors:
            if raise_exception:
                raise RuntimeError("; ".join(errors))
            return False, errors

        return True, []

    @staticmethod
    def validate_adjustment_factors(
        data: pd.DataFrame,
        raise_exception: bool = False
    ) -> tuple[bool, list[str]]:
        """
        验证复权因子数据。

        Args:
            data: 复权因子 DataFrame，应含 qfq_factor / hfq_factor 列
            raise_exception: True 时验证失败抛出 RuntimeError

        Returns:
            (is_valid, errors)
        """
        errors: list[str] = []

        required_cols = ['qfq_factor', 'hfq_factor']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"缺少必需列: {missing_cols}")
            if raise_exception:
                raise RuntimeError("; ".join(errors))
            return False, errors

        for col in required_cols:
            neg = (data[col] < 0).sum()
            if neg > 0:
                errors.append(f"列 {col} 存在 {neg} 个负值")

        dup_df = DataValidator.check_duplicates(data)
        if not dup_df.empty:
            errors.append(f"发现 {len(dup_df)} 行重复索引")

        if errors:
            if raise_exception:
                raise RuntimeError("; ".join(errors))
            return False, errors

        return True, []

    @staticmethod
    def validate_market_value(
        data: pd.DataFrame,
        raise_exception: bool = False
    ) -> tuple[bool, list[str]]:
        """
        验证市值数据。

        Args:
            data: 市值 DataFrame，应含 total_market_cap / circulating_market_cap 列
            raise_exception: True 时验证失败抛出 RuntimeError

        Returns:
            (is_valid, errors)
        """
        errors: list[str] = []

        if 'total_market_cap' in data.columns:
            neg = (data['total_market_cap'] < 0).sum()
            if neg > 0:
                errors.append(f"发现 {neg} 行总市值为负")

        if 'circulating_market_cap' in data.columns:
            neg = (data['circulating_market_cap'] < 0).sum()
            if neg > 0:
                errors.append(f"发现 {neg} 行流通市值为负")

        if 'total_market_cap' in data.columns and 'circulating_market_cap' in data.columns:
            invalid = (data['circulating_market_cap'] > data['total_market_cap']).sum()
            if invalid > 0:
                errors.append(f"发现 {invalid} 行流通市值 > 总市值")

        dup_df = DataValidator.check_duplicates(data)
        if not dup_df.empty:
            errors.append(f"发现 {len(dup_df)} 行重复索引")

        if errors:
            if raise_exception:
                raise RuntimeError("; ".join(errors))
            return False, errors

        return True, []

    # ==================== 数据对比与合并 ====================

    @staticmethod
    def compare_and_merge(
        new_data: pd.DataFrame,
        existing_data: pd.DataFrame,
        data_type: str = 'unknown',
        logger: Optional[logging.Logger] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        将新数据与已有数据对比并合并。

        合并策略：新数据优先覆盖重叠日期行，仅旧数据有的日期行予以保留。
            merged = old[only_old_dates] ∪ new[all_dates]

        时区处理：内部自动统一为 naive datetime（移除 tzinfo），调用方无需预处理。

        Args:
            new_data:      新获取的数据（DatetimeIndex）
            existing_data: 已存储的数据（DatetimeIndex）
            data_type:     数据类型标识，仅用于日志（'ohlcv' / 'adjustment' / ...）
            logger:        可选外部 logger；None 时使用模块级 logger

        Returns:
            (merged_df, diff_summary)

            merged_df: 合并后的完整 DataFrame，索引已排序且无重复
            diff_summary: {
                "new_rows":        int,        # new_data 中新增的日期数（不在 existing 中）
                "overlap_rows":    int,        # 重叠日期数
                "updated_cells":   int,        # 重叠行中值发生变化的单元格数
                "kept_old_rows":   int,        # 仅旧数据有的行数（被保留）
                "changed_dates":   list[str],  # 值变化的日期（最多 20 条）
                "column_mismatch": bool        # 列名是否不一致
            }
        """
        log = logger or globals()['logger']

        # ---------- 统一时区（防御性） ----------
        def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
            if df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)
            return df

        new_data = _strip_tz(new_data)
        existing_data = _strip_tz(existing_data)

        # ---------- 列名一致性检查 ----------
        new_cols = set(new_data.columns)
        old_cols = set(existing_data.columns)
        column_mismatch = new_cols != old_cols
        if column_mismatch:
            log.warning(
                "[%s] 列名不一致 — 新增列: %s，缺失列: %s",
                data_type, new_cols - old_cols, old_cols - new_cols
            )

        # ---------- 计算日期集合 ----------
        new_dates = set(new_data.index)
        old_dates = set(existing_data.index)

        only_old_dates = old_dates - new_dates
        overlap_dates = old_dates & new_dates
        only_new_dates = new_dates - old_dates

        # ---------- 统计重叠行的 cell 差异 ----------
        updated_cells = 0
        changed_dates: list[str] = []

        if overlap_dates:
            overlap_idx = sorted(overlap_dates)
            # 取公共列（排除 symbol 列）
            common_cols = [
                c for c in new_data.columns
                if c in existing_data.columns and c != 'symbol'
            ]
            if common_cols:
                new_overlap = new_data.loc[overlap_idx, common_cols]
                old_overlap = existing_data.loc[overlap_idx, common_cols]

                # 浮点安全比较
                try:
                    diff_mask = ~np.isclose(
                        new_overlap.values.astype(float),
                        old_overlap.values.astype(float),
                        equal_nan=True,
                        rtol=1e-9, atol=1e-12
                    )
                except (TypeError, ValueError):
                    # 含非数值列时退回到等值比较
                    diff_mask = (new_overlap != old_overlap).values

                updated_cells = int(diff_mask.sum())
                changed_row_mask = diff_mask.any(axis=1)
                changed_dates = [
                    pd.Timestamp(d).strftime('%Y-%m-%d')
                    for d in new_overlap.index[changed_row_mask]
                ][:20]

        # ---------- 执行合并 ----------
        old_only_df = existing_data.loc[sorted(only_old_dates)] if only_old_dates else existing_data.iloc[0:0]
        merged = pd.concat([old_only_df, new_data]).sort_index()
        # 保险去重（理论上已无重叠）
        merged = merged[~merged.index.duplicated(keep='last')]

        diff_summary = {
            "new_rows":        len(only_new_dates),
            "overlap_rows":    len(overlap_dates),
            "updated_cells":   updated_cells,
            "kept_old_rows":   len(only_old_dates),
            "changed_dates":   changed_dates,
            "column_mismatch": column_mismatch,
        }

        log.debug(
            "[%s] compare_and_merge 完成: 新增 %d 行, 重叠 %d 行 (变化 %d 格), 保留旧 %d 行",
            data_type,
            diff_summary["new_rows"],
            diff_summary["overlap_rows"],
            diff_summary["updated_cells"],
            diff_summary["kept_old_rows"],
        )

        return merged, diff_summary
