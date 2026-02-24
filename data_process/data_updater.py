"""
数据更新模块 - 增量更新层

职责：
- 为 DataRepository 中的所有股票追加新数据（日常增量更新）
- 调用 DataValidator 完成基础验证和新旧数据对比
- 连贯性检查：日期连贯（无交易日空缺）、hfq_factor 单调性、列结构一致
- 结构化的更新结果报告（UpdateResult）

依赖关系：
    HybridDataProvider ──► DataUpdater ◄── DataValidator
                                │
                                ▼
                         DataRepository

使用示例：
    from data_process.repository   import DataRepository
    from data_process.data_provider import HybridDataProvider
    from data_process.data_updater  import DataUpdater

    repo     = DataRepository("data/repository")
    provider = HybridDataProvider()
    updater  = DataUpdater(repo, provider)

    # 更新单只股票
    result = updater.update_symbol("600519", data_type="ohlcv")

    # 更新单只股票（只下载近3年数据以节省时间）
    result = updater.update_symbol("600519", start_date="2022-01-01", data_type="ohlcv")

    # 批量更新全部股票、全部类型
    results = updater.update_all()

    # 批量更新（只下载近30天，加速日常增量）
    results = updater.update_all(start_date="2026-01-01")
"""

import logging
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from data_process.data_validator import DataValidator
from data_process.repository import DataRepository


# ==================== 结果数据类 ====================

@dataclass
class UpdateResult:
    """单次增量更新的结构化结果。"""

    symbol: str
    data_type: str
    success: bool

    # 数据量统计
    new_rows: int = 0          # 新增行数（不在旧数据中的日期）
    overlap_rows: int = 0      # 重叠日期行数
    updated_cells: int = 0     # 重叠行中值发生变化的单元格数

    # 连贯性检查
    date_continuous: bool = True
    missing_dates: list[str] = field(default_factory=list)   # 交易日空缺列表

    hfq_monotonic: bool = True
    hfq_violations: list[str] = field(default_factory=list)  # hfq_factor 下降的日期

    column_consistent: bool = True

    # 其他信息
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        lines = [f"{status} [{self.data_type}] {self.symbol}"]
        if self.success:
            lines.append(
                f"   新增 {self.new_rows} 行  重叠 {self.overlap_rows} 行"
                f"  变化 {self.updated_cells} 格"
            )
            if not self.date_continuous:
                lines.append(f"   ⚠️  日期不连贯，缺失交易日: {self.missing_dates[:5]}")
            if not self.hfq_monotonic:
                lines.append(f"   ⚠️  hfq_factor 单调性异常: {self.hfq_violations[:5]}")
            if not self.column_consistent:
                lines.append("   ⚠️  列结构不一致")
            for w in self.warnings:
                lines.append(f"   ℹ️  {w}")
        else:
            lines.append(f"   错误: {self.error}")
        return "\n".join(lines)


# ==================== 更新器主类 ====================

# 支持的数据类型（derived 由特征工程独立维护，不在此处更新）
_SUPPORTED_TYPES = ('ohlcv', 'adjustment', 'market_value', 'financial')


class DataUpdater:
    """
    增量数据更新器。

    对已存储在 DataRepository 中的股票，拉取最新数据并追加，同时：
    - 通过 DataValidator 完成基础字段验证
    - 通过 DataValidator.compare_and_merge 统计新旧数据差异
    - 检查日期连贯性、hfq_factor 单调性、列结构一致性
    """

    def __init__(
        self,
        repository: DataRepository,
        provider,
        validator: Optional[DataValidator] = None
    ):
        """
        Args:
            repository: DataRepository 实例（存储层）
            provider:   数据提供者（HybridDataProvider 或兼容接口）
            validator:  DataValidator 实例（可选，默认自动创建）
        """
        self.repo = repository
        self.provider = provider
        self.validator = validator or DataValidator()
        self.logger = logging.getLogger('DataUpdater')

    # ==================== 公开方法 ====================

    def update_symbol(
        self,
        symbol: str,
        start_date: str = '2000-01-01',
        end_date: Optional[str] = None,
        data_type: str = 'ohlcv',
        dry_run: bool = False
    ) -> UpdateResult:
        """
        对单只股票的指定数据类型执行增量更新。

        每次从 start_date 到 end_date 全量下载，通过 compare_and_merge
        自动识别新增行、变化行，无需预先查询已有数据的日期范围。

        流程：
        1. 从 provider 全量下载 [start_date, end_date] 的数据
        2. DataValidator 基础验证
        3. 加载已有数据；若有则执行三项连贯性检查 + 合并
        4. dry_run=False 时写入合并结果

        Args:
            symbol:     股票代码
            start_date: 下载起始日期（默认 '2000-01-01'）
            end_date:   截止日期（默认今天）
            data_type:  数据类型（'ohlcv' / 'adjustment' / 'market_value' / 'financial'）
            dry_run:    True 时只校验、不写入

        Returns:
            UpdateResult
        """
        if data_type not in _SUPPORTED_TYPES:
            return UpdateResult(
                symbol=symbol, data_type=data_type, success=False,
                error=f"不支持的数据类型: {data_type}，支持: {_SUPPORTED_TYPES}"
            )

        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        result = UpdateResult(symbol=symbol, data_type=data_type, success=False)

        try:
            # ---- 1. 下载新数据 ----
            new_data = self._fetch_data(symbol, start_date, end_date, data_type)
            if new_data is None or new_data.empty:
                result.success = True
                result.warnings.append(f"provider 返回空数据（{start_date} ~ {end_date}）")
                return result

            # ---- 2. 基础验证 ----
            valid, errors = self._run_basic_validation(new_data, data_type)
            if not valid:
                result.error = f"基础验证失败: {'; '.join(errors)}"
                return result

            # ---- 3. 加载已有数据并做连贯性检查 + 合并 ----
            existing_data = self._load_existing(symbol, data_type)

            if existing_data is not None and not existing_data.empty:
                # 3a. 列结构一致性
                result.column_consistent = self._check_column_consistency(
                    new_data, existing_data
                )
                if not result.column_consistent:
                    result.warnings.append("新旧数据列结构不一致，已强制合并")

                # 3b. 日期连贯性（financial 季报全量返回，跳过）
                if data_type != 'financial':
                    continuous, missing = self._check_date_continuity(
                        new_data, existing_data
                    )
                    result.date_continuous = continuous
                    result.missing_dates = missing
                    if not continuous:
                        result.warnings.append(
                            f"日期不连贯，缺失 {len(missing)} 个交易日"
                        )

                # 3c. 对比合并（新数据优先覆盖重叠行）
                merged, diff = DataValidator.compare_and_merge(
                    new_data, existing_data,
                    data_type=data_type, logger=self.logger
                )
                result.new_rows = diff["new_rows"]
                result.overlap_rows = diff["overlap_rows"]
                result.updated_cells = diff["updated_cells"]

                if diff["column_mismatch"]:
                    result.column_consistent = False

                # 3d. hfq_factor 单调性（仅 adjustment）
                if data_type == 'adjustment' and 'hfq_factor' in merged.columns:
                    mono, violations = self._check_hfq_monotonicity(merged)
                    result.hfq_monotonic = mono
                    result.hfq_violations = violations
                    if not mono:
                        result.warnings.append(
                            f"hfq_factor 出现 {len(violations)} 处单调性异常"
                        )

                data_to_save = merged
            else:
                # 首次保存，无旧数据
                result.new_rows = len(new_data)
                data_to_save = new_data

            # ---- 4. 写入 ----
            if not dry_run:
                self._save_data(symbol, data_to_save, data_type)

            result.success = True
            self.logger.info(
                "[%s][%s] 更新完成：新增 %d 行, 重叠 %d 行, 变化 %d 格",
                symbol, data_type,
                result.new_rows, result.overlap_rows, result.updated_cells
            )

        except Exception as exc:
            result.error = str(exc)
            self.logger.error("[%s][%s] 更新失败: %s", symbol, data_type, exc, exc_info=True)

        return result

    def update_all(
        self,
        start_date: str = '2000-01-01',
        end_date: Optional[str] = None,
        data_types: Optional[list[str]] = None,
        symbols: Optional[list[str]] = None,
        skip_on_error: bool = True
    ) -> dict[str, UpdateResult]:
        """
        批量更新 DataRepository 中所有（或指定）股票的所有（或指定）数据类型。

        每只股票每种数据类型均从 start_date 全量下载，由 compare_and_merge 决定
        实际新增/变化内容，无需预先查询已有数据范围。

        Args:
            start_date:     下载起始日期（默认 '2000-01-01'；可缩短以减少下载量）
            end_date:       截止日期（默认今天）
            data_types:     要更新的数据类型列表（默认全部支持类型）
            symbols:        要更新的股票列表（默认从 ohlcv 目录读取全部）
            skip_on_error:  遇到单只股票失败时是否继续（True=继续，False=抛出异常）

        Returns:
            dict[str, UpdateResult]，key 格式为 "{symbol}_{data_type}"
        """
        if data_types is None:
            data_types = list(_SUPPORTED_TYPES)
        if symbols is None:
            symbols = self.repo.list_symbols('ohlcv')
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        results: dict[str, UpdateResult] = {}
        total = len(symbols) * len(data_types)
        done = 0

        self.logger.info(
            "开始批量更新：%d 只股票 × %d 种数据类型 = %d 任务",
            len(symbols), len(data_types), total
        )

        for symbol in symbols:
            for dtype in data_types:
                key = f"{symbol}_{dtype}"
                try:
                    result = self.update_symbol(symbol, start_date=start_date, end_date=end_date, data_type=dtype)
                    results[key] = result
                except Exception as exc:
                    result = UpdateResult(
                        symbol=symbol, data_type=dtype, success=False,
                        error=str(exc)
                    )
                    results[key] = result
                    self.logger.error("[%s][%s] 未捕获异常: %s", symbol, dtype, exc)
                    if not skip_on_error:
                        raise

                done += 1
                if done % 20 == 0 or done == total:
                    self.logger.info("进度 %d/%d", done, total)

        # 汇总日志
        success_count = sum(1 for r in results.values() if r.success)
        self.logger.info(
            "批量更新完成：%d/%d 成功", success_count, total
        )

        return results

    # ==================== 连贯性检查（私有） ====================

    def _check_date_continuity(
        self,
        new_data: pd.DataFrame,
        existing_data: pd.DataFrame
    ) -> tuple[bool, list[str]]:
        """
        检查新旧数据之间是否存在交易日空缺。

        若新数据的起始日期 <= 旧数据的最终日期（有重叠），视为连续。
        否则获取 [old_end+1, new_start-1] 区间内的交易日，若有则不连续。

        降级方案：AkShare 不可用时使用自然工作日（bdate_range）近似判断。

        Returns:
            (is_continuous, missing_trading_days_list)
        """
        # 统一为 naive datetime
        def _naive_idx(df):
            idx = df.index
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            return idx

        new_idx = _naive_idx(new_data)
        old_idx = _naive_idx(existing_data)

        old_end = old_idx.max()
        new_start = new_idx.min()

        # 有重叠 → 连续
        if new_start <= old_end:
            return True, []

        gap_start = old_end + pd.Timedelta(days=1)
        gap_end = new_start - pd.Timedelta(days=1)

        if gap_start > gap_end:
            return True, []

        # 尝试用 AkShare 交易日历
        try:
            import akshare as ak
            cal = ak.tool_trade_date_hist_sina()
            # 返回 DataFrame，日期列名可能是 'trade_date' 或第一列
            if isinstance(cal, pd.DataFrame):
                date_col = cal.columns[0]
                trading_days = pd.to_datetime(cal[date_col])
            else:
                trading_days = pd.to_datetime(cal)

            missing = trading_days[
                (trading_days >= gap_start) & (trading_days <= gap_end)
            ]
            missing_list = [d.strftime('%Y-%m-%d') for d in missing]
        except Exception as e:
            self.logger.warning("AkShare 交易日历获取失败，改用 bdate_range 近似: %s", e)
            missing = pd.bdate_range(start=gap_start, end=gap_end)
            missing_list = [d.strftime('%Y-%m-%d') for d in missing]

        if missing_list:
            return False, missing_list
        return True, []

    def _check_hfq_monotonicity(
        self,
        merged_data: pd.DataFrame
    ) -> tuple[bool, list[str]]:
        """
        检查 hfq_factor 是否单调不降（后复权因子应随时间单调递增）。

        使用浮点容差 1e-8，避免微小精度误差误报。

        Returns:
            (is_monotonic, violation_dates)
        """
        if 'hfq_factor' not in merged_data.columns:
            return True, []

        series = merged_data['hfq_factor'].sort_index()
        diff = series.diff()
        violations_mask = diff < -1e-8
        violation_dates = [
            d.strftime('%Y-%m-%d')
            for d in series.index[violations_mask]
        ]

        if violation_dates:
            return False, violation_dates
        return True, []

    @staticmethod
    def _check_column_consistency(
        new_data: pd.DataFrame,
        existing_data: pd.DataFrame
    ) -> bool:
        """检查新旧数据列名是否一致。"""
        return set(new_data.columns) == set(existing_data.columns)

    # ==================== 内部路由辅助（私有） ====================

    def _fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data_type: str
    ) -> Optional[pd.DataFrame]:
        """
        按 data_type 路由到对应的 provider 方法下载数据。

        adjustment 特殊处理
        -------------------
        复权因子是「从查询起始日归一化为 1.0」的累乘序列。
        若已有历史数据，必须从其最早日期重新下载，保证新旧数据在同一
        归一化基准下，避免拼接时出现比例断层（如 5.57 → 1.01 的跳变）。
        """
        try:
            if data_type == 'ohlcv':
                return self.provider.get_kline_data(symbol, start_date, end_date)
            elif data_type == 'adjustment':
                # 从已有数据的最早日期重新下载，保持因子连贯性
                adj_start = start_date
                existing = self._load_existing(symbol, 'adjustment')
                if existing is not None and not existing.empty:
                    idx = existing.index
                    if idx.tz is not None:
                        idx = idx.tz_localize(None)
                    earliest = idx.min()
                    adj_start = min(
                        pd.to_datetime(start_date),
                        pd.Timestamp(earliest)
                    ).strftime('%Y-%m-%d')
                    if adj_start != start_date:
                        self.logger.debug(
                            "[%s] adjustment 回溯至已有数据最早日期 %s", symbol, adj_start
                        )
                return self.provider.get_adjustment_factors(symbol, adj_start, end_date)
            elif data_type == 'market_value':
                return self.provider.get_market_value_data(symbol, start_date, end_date)
            elif data_type == 'financial':
                # AkShare 财务数据不支持日期参数，全量返回
                return self.provider.get_financial_indicators(symbol)
            else:
                self.logger.error("未知数据类型: %s", data_type)
                return None
        except Exception as e:
            self.logger.error("[%s][%s] 数据下载失败: %s", symbol, data_type, e)
            raise

    def _run_basic_validation(
        self,
        data: pd.DataFrame,
        data_type: str
    ) -> tuple[bool, list[str]]:
        """按 data_type 路由到对应的 DataValidator 方法。"""
        if data_type == 'ohlcv':
            return DataValidator.validate_ohlcv(data)
        elif data_type == 'adjustment':
            return DataValidator.validate_adjustment_factors(data)
        elif data_type == 'market_value':
            return DataValidator.validate_market_value(data)
        elif data_type == 'financial':
            # 财务数据无通用字段约束，仅检查重复索引
            dup_df = DataValidator.check_duplicates(data)
            if not dup_df.empty:
                return False, [f"发现 {len(dup_df)} 行重复索引"]
            return True, []
        return True, []

    def _load_existing(
        self,
        symbol: str,
        data_type: str
    ) -> Optional[pd.DataFrame]:
        """加载已有数据，文件不存在时返回 None。"""
        try:
            if data_type == 'ohlcv':
                return self.repo.load_ohlcv_data(symbol)
            elif data_type == 'adjustment':
                return self.repo.load_adjustment_factors(symbol)
            elif data_type == 'market_value':
                return self.repo.load_market_value_data(symbol)
            elif data_type == 'financial':
                return self.repo.load_financial_data(symbol)
        except RuntimeError:
            # 数据文件尚不存在
            return None
        return None

    def _save_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        data_type: str
    ) -> None:
        """
        将已合并好的数据写入 DataRepository（纯覆盖）。

        时区处理由 Repository._strip_tz 统一负责，此处无需预处理。
        """
        if data_type == 'ohlcv':
            self.repo.save_ohlcv_data(symbol, data)
        elif data_type == 'adjustment':
            self.repo.save_adjustment_factors(symbol, data)
        elif data_type == 'market_value':
            self.repo.save_market_value_data(symbol, data)
        elif data_type == 'financial':
            self.repo.save_financial_data(symbol, data)
