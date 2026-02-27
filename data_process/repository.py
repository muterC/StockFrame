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

设计原则：
- save_* 方法均为**纯覆盖写入**，不做内部合并。
  合并/去重逻辑由上层 DataUpdater.compare_and_merge 在写入前完成。
- load_* 方法支持可选的日期范围过滤。

特性：
- 多数据类型支持（OHLCV + 复权因子 + 财务 + 市值 + 衍生特征）
- 结构化存储（清晰的目录组织）
- 元数据管理（数据清单、更新时间、完整性校验）
- 易于扩展（预留分钟线数据结构）
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
import warnings

from data_process.data_validator import DataValidator


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

    # ==================== OHLCV数据方法 ====================
    def save_ohlcv_data(
        self,
        symbol: str,
        data: pd.DataFrame,
    ) -> Path:
        """
        保存OHLCV数据到market/daily/ohlcv（纯覆盖写入）。

        合并逻辑由调用方（DataUpdater.compare_and_merge）在写入前完成，
        此方法只负责验证 + 写文件。

        Args:
            symbol: 股票代码
            data: OHLCV数据（需包含date索引和open,high,low,close,volume列）

        Returns:
            保存的目录路径
        """
        if data.empty:
            raise RuntimeError("不能保存空数据")

        # 验证数据完整性、逻辑正确性，以及检查重复数据
        valid, errors = DataValidator.validate_ohlcv(data, raise_exception=False)
        if not valid:
            raise RuntimeError(f"数据验证失败: {'; '.join(errors)}")

        symbol_dir = self.ohlcv_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        self._strip_tz(data).to_csv(symbol_dir / "data.csv")

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
    def save_adjustment_factors(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Path:
        """
        保存复权因子到market/daily/adjustment（纯覆盖写入）。

        Args:
            symbol: 股票代码
            data: 复权因子数据（需包含qfq_factor/hfq_factor列）

        Returns:
            保存的文件路径
        """
        if data.empty:
            raise RuntimeError("不能保存空数据")

        # 验证数据完整性和逻辑正确性
        valid, errors = DataValidator.validate_adjustment_factors(data, raise_exception=False)
        if not valid:
            raise RuntimeError(f"数据验证失败: {'; '.join(errors)}")

        symbol_dir = self.adjustment_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        self._strip_tz(data).to_csv(symbol_dir / "data.csv")

        return symbol_dir / "data.csv"

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
        保存财务指标到fundamental/financial_indicators（纯覆盖写入）。

        Args:
            symbol: 股票代码
            data: 财务指标数据（date为索引，指标为列）

        Returns:
            保存的文件路径
        """
        if data.empty:
            raise RuntimeError("不能保存空数据")

        symbol_dir = self.financial_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        self._strip_tz(data).to_csv(symbol_dir / "data.csv")

        return symbol_dir / "data.csv"

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
    def save_market_value_data(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Path:
        """
        保存市值数据到fundamental/market_value（纯覆盖写入）。

        Args:
            symbol: 股票代码
            data: 市值数据（date为索引，列可包含 total_market_cap、circulating_market_cap、pe 等，
                    所有列均为可选，按需传入即可）

        Returns:
            保存的文件路径
        """
        if data.empty:
            raise RuntimeError("不能保存空数据")

        # 验证数据完整性和逻辑正确性
        valid, errors = DataValidator.validate_market_value(data, raise_exception=False)
        if not valid:
            raise RuntimeError(f"数据验证失败: {'; '.join(errors)}")

        symbol_dir = self.mktcap_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        self._strip_tz(data).to_csv(symbol_dir / "data.csv")

        return symbol_dir / "data.csv"

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

    # ==================== 行业分类方法 ====================

    def save_industry_classification(self, data: pd.DataFrame) -> Path:
        """
        保存行业分类数据到 reference/industry_classification.csv（纯覆盖写入）。

        Args:
            data: 行业分类 DataFrame，要求包含以下列：
                - symbol          : 股票代码（6位，如 '000001'）
                - sw1_name        : 申万一级行业名称
                - sw2_name        : 申万二级行业名称
                - sw3_name        : 申万三级行业名称
                - sw1_code        : 申万一级行业代码（可选）
                - sw2_code        : 申万二级行业代码（可选）
                - sw3_code        : 申万三级行业代码（可选）
                索引无要求，symbol 为普通列。

        Returns:
            保存的文件路径
        """
        if data.empty:
            raise RuntimeError("不能保存空数据")

        required_cols = ['symbol', 'sw1_name', 'sw2_name', 'sw3_name']
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise RuntimeError(f"缺少必需列: {missing}")

        file_path = self.reference_dir / "industry_classification.csv"
        data.to_csv(file_path, index=False)
        return file_path

    def load_industry_classification(
        self,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        加载行业分类数据。

        Args:
            symbols: 指定股票代码列表；None 表示返回全部

        Returns:
            行业分类 DataFrame，symbol 为普通列
        """
        file_path = self.reference_dir / "industry_classification.csv"
        if not file_path.exists():
            raise RuntimeError("行业分类数据不存在，请先运行导入流程")

        data = pd.read_csv(file_path, dtype={'symbol': str})

        if symbols is not None:
            data = data[data['symbol'].isin(symbols)].reset_index(drop=True)

        return data

    # ==================== 私有方法 ====================

    @staticmethod
    def _strip_tz(data: pd.DataFrame) -> pd.DataFrame:
        """
        移除 DatetimeIndex 的时区信息，返回 naive datetime 副本。

        若索引已是 naive datetime，原样返回（不复制）。
        """
        if data.index.tz is not None:
            data = data.copy()
            data.index = data.index.tz_localize(None)
        return data

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
