"""
数据仓库模块 - 用于多维度数据的持久化存储和管理

架构设计：
- 市场数据 (market/)：OHLCV、复权因子、派生特征
- 基本面数据 (fundamental/)：财务指标、市值数据
- 参考数据 (reference/)：行业分类、交易日历、股票信息

存储策略：
- 所有数据：单文件存储（{symbol}/data.csv 格式）
- 数据格式：CSV 格式，日期作为索引（第一列），指标作为列
- 元数据：完整的数据追踪和校验系统

特性：
- 多数据类型支持（OHLCV + 复权因子 + 财务 + 市值 + 衍生特征）
- 结构化存储（清晰的目录组织）
- 增量更新友好（支持数据合并）
- 元数据管理（数据清单、更新时间、完整性校验）
- 易于扩展（预留分钟线数据结构）
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import json
import warnings


class DataRepository:
    """
    数据仓库类

    目录结构：
    data/repository/
    ├── index.json                              # 全局索引
    ├── metadata/                               # 集中元数据管理
    │   ├── symbols_registry.json              # 股票注册表
    │   └── update_history.json                # 更新历史
    │
    ├── market/                                 # 市场数据
    │   ├── daily/                              # 日线数据
    │   │   ├── ohlcv/                          # OHLCV原始数据
    │   │   │   ├── 600519/
    │   │   │   │   ├── data.csv               # 单个CSV文件包含所有日期
    │   │   │   │   └── metadata.json
    │   │   │   └── 601318/
    │   │   │
    │   │   ├── adjustment/                     # 复权因子
    │   │   │   ├── 600519/
    │   │   │   │   └── data.csv
    │   │   │   └── 601318/
    │   │   │
    │   │   └── derived/                        # 派生特征
    │   │       ├── returns/                    # 收益率
    │   │       ├── volatility/                 # 波动率
    │   │       └── momentum/                   # 动量指标
    │   │
    │   └── minute/                             # 分钟线数据（未来扩展）
    │       ├── 1min/
    │       ├── 5min/
    │       └── 15min/
    │
    ├── fundamental/                            # 基本面数据
    │   ├── financial_indicators/               # 财务指标
    │   │   ├── 600519/
    │   │   │   └── data.csv
    │   │   └── 601318/
    │   │
    │   └── market_value/                       # 市值数据
    │       ├── 600519/
    │       │   └── data.csv
    │       └── 601318/
    │
    └── reference/                              # 参考数据
        ├── industry_classification.csv         # 行业分类
        ├── trading_calendar.csv                # 交易日历
        └── stock_info.csv                      # 股票基本信息
    """

    def __init__(
        self,
        base_dir: Union[str, Path] = "data/repository",
        default_format: str = "csv"
    ):
        """
        Args:
            base_dir: 数据仓库根目录
            default_format: 默认存储格式 (csv/parquet)
        """
        self.base_dir = Path(base_dir)
        self.default_format = default_format

        # 创建主目录结构
        self.market_dir = self.base_dir / "market"
        self.fundamental_dir = self.base_dir / "fundamental"
        self.reference_dir = self.base_dir / "reference"
        self.metadata_dir = self.base_dir / "metadata"

        # 市场数据子目录
        self.ohlcv_dir = self.market_dir / "daily" / "ohlcv"
        self.adjustment_dir = self.market_dir / "daily" / "adjustment"
        self.derived_dir = self.market_dir / "daily" / "derived"
        self.minute_dir = self.market_dir / "minute"

        # 基本面数据子目录
        self.financial_dir = self.fundamental_dir / "financial_indicators"
        self.mktcap_dir = self.fundamental_dir / "market_value"

        # 创建所有目录
        for dir_path in [
            self.ohlcv_dir, self.adjustment_dir, self.derived_dir,
            self.minute_dir, self.financial_dir, self.mktcap_dir,
            self.reference_dir, self.metadata_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 索引文件路径
        self.index_file = self.base_dir / "index.json"
        self.symbols_registry_file = self.metadata_dir / "symbols_registry.json"
        self.update_history_file = self.metadata_dir / "update_history.json"

        # 加载索引
        self.index = self._load_index()
        self.symbols_registry = self._load_symbols_registry()

    def check_duplicates(self, data: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """
        检查重复数据

        Args:
            data: 数据

        Returns:
            Tuple[bool, pd.DataFrame]: (是否有重复, 重复的行)
        """
        duplicates = data[data.index.duplicated(keep=False)]
        has_duplicates = len(duplicates) > 0

        return has_duplicates, duplicates


    # ==================== OHLCV数据方法 ====================
    def validate_ohlcv(self, data: pd.DataFrame, raise_exception: bool = True) -> bool:
        """
        验证OHLCV数据的完整性和逻辑正确性

        Args:
            data: OHLCV数据
            raise_exception: 是否在验证失败时抛出异常

        Returns:
            bool: 验证是否通过

        Raises:
            DataValidationException: 数据验证失败
        """
        errors = []

        # 1. 检查必需列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")

        if errors:
            if raise_exception:
                raise RuntimeError("; ".join(errors))
            return False

        # 2. 检查数据类型
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"Column {col} is not numeric")

        # 3. 检查缺失值
        null_counts = data[required_cols].isnull().sum()
        if null_counts.any():
            errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

        # 4. 检查价格逻辑关系
        # high >= low
        invalid_high_low = (data['high'] < data['low']).sum()
        if invalid_high_low > 0:
            errors.append(f"Found {invalid_high_low} rows where high < low")

        # high >= open
        invalid_high_open = (data['high'] < data['open']).sum()
        if invalid_high_open > 0:
            errors.append(f"Found {invalid_high_open} rows where high < open")

        # high >= close
        invalid_high_close = (data['high'] < data['close']).sum()
        if invalid_high_close > 0:
            errors.append(f"Found {invalid_high_close} rows where high < close")

        # low <= open
        invalid_low_open = (data['low'] > data['open']).sum()
        if invalid_low_open > 0:
            errors.append(f"Found {invalid_low_open} rows where low > open")

        # low <= close
        invalid_low_close = (data['low'] > data['close']).sum()
        if invalid_low_close > 0:
            errors.append(f"Found {invalid_low_close} rows where low > close")

        # 5. 检查价格和成交量是否为负
        negative_prices = (data[['open', 'high', 'low', 'close']] < 0).any(axis=1).sum()
        if negative_prices > 0:
            errors.append(f"Found {negative_prices} rows with negative prices")

        negative_volume = (data['volume'] < 0).sum()
        if negative_volume > 0:
            errors.append(f"Found {negative_volume} rows with negative volume")


        # 汇总结果
        if errors:
            error_msg = "; ".join(errors)
            if raise_exception:
                raise RuntimeError(error_msg)
            return False

        return True

    def save_ohlcv_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        overwrite: bool = False
    ) -> Path:
        """
        保存OHLCV数据到market/daily/ohlcv

        Args:
            symbol: 股票代码
            data: OHLCV数据（需包含date索引和open,high,low,close,volume列）
            overwrite: 是否覆盖已存在的文件

        Returns:
            保存的文件路径
        """
        if data.empty:
            raise RuntimeError("不能保存空数据")

        # 验证数据完整性, 逻辑正确性，以及检查重复数据
        if self.validate_ohlcv(data, raise_exception=False) is False or self.check_duplicates(data)[0]:
            raise RuntimeError(f"数据验证失败")

        # 保存到单个文件
        symbol_dir = self.ohlcv_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        file_path = symbol_dir / "data.csv"

        if file_path.exists() and not overwrite:
            # 合并数据
            old_data = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # 统一时区处理：移除所有时区信息，统一使用naive datetime
            if old_data.index.tz is not None:
                old_data.index = old_data.index.tz_localize(None)
            if data.index.tz is not None:
                data_to_merge = data.copy()
                data_to_merge.index = data_to_merge.index.tz_localize(None)
            else:
                data_to_merge = data

            merged = pd.concat([old_data, data_to_merge])
            merged = merged[~merged.index.duplicated(keep='last')]
            merged.sort_index(inplace=True)
            merged.to_csv(file_path)
        else:
            # 新数据也移除时区
            data_to_save = data.copy()
            if data_to_save.index.tz is not None:
                data_to_save.index = data_to_save.index.tz_localize(None)
            data_to_save.to_csv(file_path)

        # 更新元数据
        self._update_symbol_metadata(
            symbol=symbol,
            data_type='ohlcv',
            start_date=data.index.min().strftime('%Y-%m-%d'),
            end_date=data.index.max().strftime('%Y-%m-%d'),
            rows=len(data)
        )

        return symbol_dir

    def load_ohlcv_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载OHLCV数据

        Args:
            symbol: 股票代码
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            OHLCV数据DataFrame
        """
        symbol_dir = self.ohlcv_dir / symbol
        file_path = symbol_dir / "data.csv"

        if not file_path.exists():
            raise RuntimeError(f"股票 {symbol} 没有OHLCV数据")

        # 加载数据
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        data = data[~data.index.duplicated(keep='last')]
        data.sort_index(inplace=True)

        # 过滤日期范围
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        return data

    # ==================== 复权因子方法 ====================
    def validate_adjustment_factors(self, data: pd.DataFrame, raise_exception: bool = False) -> bool:
        """
        验证复权因子数据

        Args:
            data: 复权因子数据
            raise_exception: 是否在验证失败时抛出异常

        Returns:
            bool: 验证是否通过
        """
        errors = []

        required_cols = ['qfq_factor', 'hfq_factor']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        if not errors:
            for col in required_cols:
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    errors.append(f"Found {negative_count} negative values in {col}")

        if errors:
            error_msg = "; ".join(errors)
            print(f"复权因子数据验证失败: {error_msg}")
            if raise_exception:
                raise RuntimeError(error_msg)
            return False

        return True

    def save_adjustment_factors(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Path:
        """
        保存复权因子到market/daily/adjustment

        Args:
            symbol: 股票代码
            data: 复权因子数据（需包含adj_factor列）

        Returns:
            保存的文件路径
        """
        if data.empty:
            raise RuntimeError("不能保存空数据")
        
        # 验证数据完整性和逻辑正确性
        if self.validate_adjustment_factors(data, raise_exception=False) is False or self.check_duplicates(data)[0]:
            raise RuntimeError(f"数据验证失败")

        symbol_dir = self.adjustment_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        file_path = symbol_dir / "data.csv"

        if file_path.exists():
            # 合并数据
            old_data = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # 统一时区处理
            if old_data.index.tz is not None:
                old_data.index = old_data.index.tz_localize(None)
            if data.index.tz is not None:
                data_to_merge = data.copy()
                data_to_merge.index = data_to_merge.index.tz_localize(None)
            else:
                data_to_merge = data

            merged = pd.concat([old_data, data_to_merge])
            merged = merged[~merged.index.duplicated(keep='last')]
            merged.sort_index(inplace=True)
            merged.to_csv(file_path)
        else:
            data_to_save = data.copy()
            if data_to_save.index.tz is not None:
                data_to_save.index = data_to_save.index.tz_localize(None)
            data_to_save.to_csv(file_path)

        return file_path

    def load_adjustment_factors(
        self,
        symbol: str
    ) -> pd.DataFrame:
        """
        加载复权因子

        Args:
            symbol: 股票代码

        Returns:
            复权因子DataFrame
        """
        file_path = self.adjustment_dir / symbol / "data.csv"

        if not file_path.exists():
            raise RuntimeError(f"股票 {symbol} 没有复权因子数据")

        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        return data

    # ==================== 财务指标方法 ====================

    def save_financial_data(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Path:
        """
        保存财务指标到fundamental/financial_indicators

        Args:
            symbol: 股票代码
            data: 财务指标数据（date为索引，指标为列，如 revenue, net_profit等）

        Returns:
            保存的文件路径
        """
        if data.empty:
            raise RuntimeError("不能保存空数据")

        symbol_dir = self.financial_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        file_path = symbol_dir / "data.csv"

        if file_path.exists():
            # 合并数据
            old_data = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # 统一时区处理
            if old_data.index.tz is not None:
                old_data.index = old_data.index.tz_localize(None)
            if data.index.tz is not None:
                data_to_merge = data.copy()
                data_to_merge.index = data_to_merge.index.tz_localize(None)
            else:
                data_to_merge = data

            merged = pd.concat([old_data, data_to_merge])
            merged = merged[~merged.index.duplicated(keep='last')]
            merged.sort_index(inplace=True)
            merged.to_csv(file_path)
        else:
            data_to_save = data.copy()
            if data_to_save.index.tz is not None:
                data_to_save.index = data_to_save.index.tz_localize(None)
            data_to_save.to_csv(file_path)

        return file_path

    def load_financial_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载财务指标

        Args:
            symbol: 股票代码
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            财务指标DataFrame
        """
        file_path = self.financial_dir / symbol / "data.csv"

        if not file_path.exists():
            raise RuntimeError(f"股票 {symbol} 没有财务指标数据")

        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # 过滤日期范围
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        return data

    # ==================== 市值数据方法 ====================
    def validate_market_value(self, data: pd.DataFrame, raise_exception: bool = False) -> bool:
        """
        验证市值数据

        Args:
            data: 市值数据
            raise_exception: 是否在验证失败时抛出异常

        Returns:
            bool: 验证是否通过
        """
        errors = []

        if 'total_market_cap' in data.columns:
            negative_count = (data['total_market_cap'] < 0).sum()
            if negative_count > 0:
                errors.append(f"Found {negative_count} negative total market cap values")

            huge_cap = (data['total_market_cap'] > 100000).sum()

        if 'circulating_market_cap' in data.columns:
            negative_count = (data['circulating_market_cap'] < 0).sum()
            if negative_count > 0:
                errors.append(f"Found {negative_count} negative circulating market cap values")

        # 检查流通市值应小于等于总市值
        if 'total_market_cap' in data.columns and 'circulating_market_cap' in data.columns:
            invalid = (data['circulating_market_cap'] > data['total_market_cap']).sum()
            if invalid > 0:
                errors.append(f"Found {invalid} rows where circulating cap > total cap")

        if errors:
            error_msg = "; ".join(errors)
            if raise_exception:
                raise RuntimeError(error_msg)
            return False

        return True

    def save_market_value_data(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Path:
        """
        保存市值数据到fundamental/market_value

        Args:
            symbol: 股票代码
            data: 市值数据（date为索引，包含total_market_cap, circulating_market_cap等列）

        Returns:
            保存的文件路径
        """
        if data.empty:
            raise RuntimeError("不能保存空数据")
        
        # 验证数据完整性和逻辑正确性
        if self.validate_market_value(data, raise_exception=False) is False or self.check_duplicates(data)[0]:
            raise RuntimeError(f"数据验证失败")

        symbol_dir = self.mktcap_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        file_path = symbol_dir / "data.csv"

        if file_path.exists():
            # 合并数据
            old_data = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # 统一时区处理
            if old_data.index.tz is not None:
                old_data.index = old_data.index.tz_localize(None)
            if data.index.tz is not None:
                data_to_merge = data.copy()
                data_to_merge.index = data_to_merge.index.tz_localize(None)
            else:
                data_to_merge = data

            merged = pd.concat([old_data, data_to_merge])
            merged = merged[~merged.index.duplicated(keep='last')]
            merged.sort_index(inplace=True)
            merged.to_csv(file_path)
        else:
            data_to_save = data.copy()
            if data_to_save.index.tz is not None:
                data_to_save.index = data_to_save.index.tz_localize(None)
            data_to_save.to_csv(file_path)

        return file_path

    def load_market_value_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载市值数据

        Args:
            symbol: 股票代码
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            市值数据DataFrame
        """
        file_path = self.mktcap_dir / symbol / "data.csv"

        if not file_path.exists():
            raise RuntimeError(f"股票 {symbol} 没有市值数据")

        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # 过滤日期范围
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        return data

    # ==================== 派生特征方法 ====================

    def save_derived_features(
        self,
        symbol: str,
        data: pd.DataFrame,
        feature_type: str
    ) -> Path:
        """
        保存派生特征到market/daily/derived

        Args:
            symbol: 股票代码
            data: 派生特征数据
            feature_type: 特征类型（可自定义，如 'returns', 'volatility', 'momentum', 'alpha_101' 等）

        Returns:
            保存的文件路径

        Note:
            feature_type 可以是任意字符串，系统会自动创建对应的子目录
            这使得派生特征的类型完全可扩展，不受限制
        """
        if data.empty:
            raise RuntimeError("不能保存空数据")

        # 验证feature_type是合法的目录名
        if not feature_type or '/' in feature_type or '\\' in feature_type:
            raise RuntimeError(f"特征类型名称不合法: {feature_type}")

        # 自动创建特征类型目录
        feature_dir = self.derived_dir / feature_type
        feature_dir.mkdir(parents=True, exist_ok=True)

        file_path = feature_dir / f"{symbol}.csv"

        if file_path.exists():
            # 合并数据
            old_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            merged = pd.concat([old_data, data])
            merged = merged[~merged.index.duplicated(keep='last')]
            merged.sort_index(inplace=True)
            merged.to_csv(file_path)
        else:
            data.to_csv(file_path)

        return file_path

    def load_derived_features(
        self,
        symbol: str,
        feature_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载派生特征

        Args:
            symbol: 股票代码
            feature_type: 特征类型（任意字符串，如 'returns', 'volatility', 'momentum', 'alpha_101' 等）
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            派生特征DataFrame

        Raises:
            RuntimeError: 如果指定的特征类型或股票数据不存在
        """
        file_path = self.derived_dir / feature_type / f"{symbol}.csv"

        if not file_path.exists():
            raise RuntimeError(f"股票 {symbol} 没有 {feature_type} 特征数据")

        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # 过滤日期范围
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        return data

    def list_derived_feature_types(self) -> List[str]:
        """
        列出所有可用的派生特征类型

        Returns:
            特征类型名称列表
        """
        if not self.derived_dir.exists():
            return []

        feature_types = [
            d.name for d in self.derived_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]

        return sorted(feature_types)

    # ==================== 综合方法 ====================

    def load_complete_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_derived: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        加载完整数据（OHLCV + 复权因子 + 财务指标 + 市值）
        适用于因子挖掘和回测场景

        Args:
            symbol: 股票代码
            start_date: 起始日期
            end_date: 结束日期
            include_derived: 是否包含派生特征

        Returns:
            数据字典 {
                'ohlcv': DataFrame (必需),
                'adjustment': DataFrame (必需 - 因子挖掘时需要复权),
                'financial': DataFrame (可选),
                'market_value': DataFrame (可选),
                'derived_*': DataFrame (可选 - 根据实际保存的特征类型)
            }

        Raises:
            RuntimeError: 当OHLCV或复权因子数据缺失时
        """
        result = {}

        # OHLCV（必需）
        try:
            result['ohlcv'] = self.load_ohlcv_data(symbol, start_date, end_date)
        except RuntimeError as e:
            raise RuntimeError(f"加载OHLCV数据失败: {e}")

        # 复权因子（必需 - 因子挖掘时需要）
        try:
            result['adjustment'] = self.load_adjustment_factors(symbol)
        except RuntimeError as e:
            raise RuntimeError(f"加载复权因子数据失败: {e}")

        # 财务指标（可选）
        try:
            result['financial'] = self.load_financial_data(symbol, start_date, end_date)
        except RuntimeError:
            result['financial'] = None

        # 市值数据（可选）
        try:
            result['market_value'] = self.load_market_value_data(symbol, start_date, end_date)
        except RuntimeError:
            result['market_value'] = None

        # 派生特征（可选）- 动态发现所有可用的特征类型
        if include_derived:
            available_features = self.list_derived_feature_types()
            for feature_type in available_features:
                try:
                    result[f'derived_{feature_type}'] = self.load_derived_features(
                        symbol, feature_type, start_date, end_date
                    )
                except RuntimeError:
                    result[f'derived_{feature_type}'] = None


        return result

    # ==================== 查询和管理方法 ====================

    def list_symbols(self, data_type: str = 'ohlcv') -> List[str]:
        """
        列出所有可用的股票代码

        Args:
            data_type: 数据类型 ('ohlcv', 'adjustment', 'financial', 'market_value')

        Returns:
            股票代码列表
        """
        if data_type == 'ohlcv':
            data_dir = self.ohlcv_dir
            symbols = [d.name for d in data_dir.iterdir() if d.is_dir()]
        elif data_type == 'adjustment':
            data_dir = self.adjustment_dir
            symbols = [d.name for d in data_dir.iterdir() if d.is_dir()]
        elif data_type == 'financial':
            data_dir = self.financial_dir
            symbols = [d.name for d in data_dir.iterdir() if d.is_dir()]
        elif data_type == 'market_value':
            data_dir = self.mktcap_dir
            symbols = [d.name for d in data_dir.iterdir() if d.is_dir()]
        else:
            raise RuntimeError(f"不支持的数据类型: {data_type}")

        symbols.sort()
        return symbols

    def get_data_info(self, symbol: str, data_type: str = 'ohlcv') -> Dict:
        """
        获取股票数据信息

        Args:
            symbol: 股票代码
            data_type: 数据类型

        Returns:
            数据信息字典
        """
        if data_type == 'ohlcv':
            symbol_dir = self.ohlcv_dir / symbol
            metadata_file = symbol_dir / "metadata.json"
        else:
            return {'symbol': symbol, 'data_type': data_type, 'available': False}

        if not metadata_file.exists():
            return {'symbol': symbol, 'data_type': data_type, 'available': False}

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return metadata

    def get_data_coverage(self, symbol: str) -> Dict:
        """
        获取数据覆盖情况

        Args:
            symbol: 股票代码

        Returns:
            覆盖情况字典
        """
        coverage = {
            'symbol': symbol,
            'ohlcv': self._check_data_exists(symbol, 'ohlcv'),
            'adjustment': self._check_data_exists(symbol, 'adjustment'),
            'financial': self._check_data_exists(symbol, 'financial'),
            'market_value': self._check_data_exists(symbol, 'market_value'),
            'derived_returns': self._check_data_exists(symbol, 'derived_returns'),
            'derived_volatility': self._check_data_exists(symbol, 'derived_volatility'),
            'derived_momentum': self._check_data_exists(symbol, 'derived_momentum')
        }
        return coverage

    def get_date_range(self, symbol: str, data_type: str = 'ohlcv') -> Optional[Tuple[str, str]]:
        """
        获取数据的日期范围

        Args:
            symbol: 股票代码
            data_type: 数据类型

        Returns:
            (start_date, end_date) 或 None
        """
        info = self.get_data_info(symbol, data_type)

        if not info.get('available'):
            return None

        return (info.get('start_date'), info.get('end_date'))

    def delete_data(self, symbol: str, data_type: str = 'ohlcv') -> bool:
        """
        删除股票数据

        Args:
            symbol: 股票代码
            data_type: 数据类型

        Returns:
            是否删除成功
        """
        import shutil

        if data_type == 'ohlcv':
            symbol_dir = self.ohlcv_dir / symbol
            if symbol_dir.exists():
                shutil.rmtree(symbol_dir)
                return True
        elif data_type == 'adjustment':
            symbol_dir = self.adjustment_dir / symbol
            if symbol_dir.exists():
                shutil.rmtree(symbol_dir)
                return True
        elif data_type == 'financial':
            symbol_dir = self.financial_dir / symbol
            if symbol_dir.exists():
                shutil.rmtree(symbol_dir)
                return True
        elif data_type == 'market_value':
            symbol_dir = self.mktcap_dir / symbol
            if symbol_dir.exists():
                shutil.rmtree(symbol_dir)
                return True

        return False

    def get_summary(self) -> Dict:
        """
        获取数据仓库摘要

        Returns:
            摘要信息字典
        """
        summary = {
            'version': '2.0',
            'total_symbols': 0,
            'data_types': {
                'ohlcv': len(self.list_symbols('ohlcv')),
                'adjustment': len(self.list_symbols('adjustment')),
                'financial': len(self.list_symbols('financial')),
                'market_value': len(self.list_symbols('market_value'))
            },
            'total_records': 0,
            'earliest_date': None,
            'latest_date': None
        }

        summary['total_symbols'] = summary['data_types']['ohlcv']

        # 统计记录数和日期范围
        all_dates = []
        for symbol in self.list_symbols('ohlcv'):
            try:
                data = self.load_ohlcv_data(symbol)
                summary['total_records'] += len(data)
                all_dates.extend([data.index.min(), data.index.max()])
            except Exception as e:
                continue

        if all_dates:
            summary['earliest_date'] = min(all_dates).strftime('%Y-%m-%d')
            summary['latest_date'] = max(all_dates).strftime('%Y-%m-%d')

        return summary

    # ==================== 兼容旧接口 ====================

    def save_daily_data(self, symbol: str, data: pd.DataFrame, **kwargs) -> Path:
        """兼容旧接口：保存日线数据"""
        return self.save_ohlcv_data(symbol, data, **kwargs.get('overwrite', False))

    def load_daily_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """兼容旧接口：加载日线数据"""
        return self.load_ohlcv_data(symbol, start_date, end_date)

    # ==================== 私有方法 ====================

    def _check_data_exists(self, symbol: str, data_type: str) -> bool:
        """检查数据是否存在"""
        if data_type == 'ohlcv':
            return (self.ohlcv_dir / symbol / "data.csv").exists()
        elif data_type == 'adjustment':
            return (self.adjustment_dir / symbol / "data.csv").exists()
        elif data_type == 'financial':
            return (self.financial_dir / symbol / "data.csv").exists()
        elif data_type == 'market_value':
            return (self.mktcap_dir / symbol / "data.csv").exists()
        elif data_type.startswith('derived_'):
            feature_type = data_type.replace('derived_', '')
            return (self.derived_dir / feature_type / f"{symbol}.csv").exists()
        return False

    def _update_symbol_metadata(
        self,
        symbol: str,
        data_type: str,
        start_date: str,
        end_date: str,
        rows: int
    ):
        """更新股票元数据"""
        symbol_dir = self.ohlcv_dir / symbol
        metadata_file = symbol_dir / "metadata.json"

        metadata = {
            'symbol': symbol,
            'data_type': data_type,
            'start_date': start_date,
            'end_date': end_date,
            'rows': rows,
            'updated_at': datetime.now().isoformat(),
            'available': True
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _load_index(self) -> Dict:
        """加载全局索引"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                'version': '2.0',
                'created_at': datetime.now().isoformat(),
                'symbols': {}
            }

    def _load_symbols_registry(self) -> Dict:
        """加载股票注册表"""
        if self.symbols_registry_file.exists():
            with open(self.symbols_registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}


# ==================== 便捷函数 ====================

_default_repository = None

def get_repository() -> DataRepository:
    """获取默认数据仓库实例（单例）"""
    global _default_repository
    if _default_repository is None:
        _default_repository = DataRepository()
    return _default_repository


def save_data(symbol: str, data: pd.DataFrame, **kwargs) -> Path:
    """快捷保存函数（兼容旧接口）"""
    repo = get_repository()
    return repo.save_ohlcv_data(symbol, data, **kwargs.get('overwrite', False))


def load_data(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """快捷加载函数（兼容旧接口）"""
    repo = get_repository()
    return repo.load_ohlcv_data(symbol, start_date, end_date)


def list_all_symbols() -> List[str]:
    """列出所有股票（兼容旧接口）"""
    repo = get_repository()
    return repo.list_symbols('ohlcv')
