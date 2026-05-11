"""Standalone object-facing store for free market data."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .providers import DEFAULT_PROVIDER_CLASSES, BaseProvider, HyperProvider
from .providers.base import FieldSelection
from .schema import DAILY_INDEX_COLUMNS, DEFAULT_FIELD_DESCRIPTIONS, FIELD_DESCRIPTION_COLUMNS
from .symbols import as_timestamp, normalize_codes, normalize_stock_code

ProviderName = Literal["akshare", "baostock", "yahoo", "tencent", "xueqiu", "sina", "sohu"]
ReturnFormat = Literal["long", "wide", "dict"]
DAILY_PROVIDER_PACKAGE_COLUMNS = ["open", "high", "low", "close", "qfq_factor", "hfq_factor", "price_source"]


class FreeMarketDataStore:
    """
    免费数据源和本地 TSV 数据库的唯一用户入口。

    日线数据优先读取本地 TSV 缓存，缺口自动通过 provider 补齐并入库；
    分钟级和实时行情直接调用 provider，不写入本地库。
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "data/free_market_db",
        providers: Sequence[str] = ("akshare", "baostock", "yahoo", "tencent", "xueqiu", "sina", "sohu"),
        stock_codes: Optional[Sequence[str]] = None,
        history_years: int = 1,
        auto_initialize: bool = True,
        auto_warmup: bool = True,
        provider_instances: Optional[Sequence[BaseProvider]] = None,
        provider: Optional[HyperProvider] = None,
        daily_fields: FieldSelection = None,
        minute_fields: FieldSelection = None,
        realtime_fields: FieldSelection = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.providers = tuple(str(provider).lower() for provider in providers)
        self.history_years = history_years

        self.daily_dir = self.db_path / "daily"
        self.meta_dir = self.db_path / "_meta"
        self.field_description_path = self.meta_dir / "field_descriptions.tsv"
        self._field_fetchers: Dict[str, Callable[..., pd.DataFrame]] = {}
        provider_map: Dict[str, BaseProvider] = {name: cls() for name, cls in DEFAULT_PROVIDER_CLASSES.items()}
        self.provider = provider or HyperProvider(
            provider_map=provider_map,
            provider_order=self.providers,
            daily_fields=daily_fields,
        )

        for provider in provider_instances or []:
            self.register_provider(provider)

        if auto_initialize:
            self.initialize_database()

        if auto_warmup and stock_codes:
            end_date = pd.Timestamp.today().normalize()
            start_date = end_date - pd.DateOffset(years=history_years)
            try:
                self.get_daily(stock_codes, start_date, end_date)
            except Exception as exc:
                warnings.warn(
                    f"初始化预热近 {history_years} 年日线数据失败，后续请求会继续按需拉取: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def register_provider(self, provider: BaseProvider, prefer_first: bool = False) -> "FreeMarketDataStore":
        """注册一个 provider 实例，便于后续接入新的免费数据源。"""
        name = str(getattr(provider, "name", "")).lower()
        if not name:
            raise ValueError("provider 必须提供 name 属性。")
        self.provider.register_provider(provider, prefer_first=prefer_first)
        if name not in self.providers:
            self.providers = (name, *self.providers) if prefer_first else (*self.providers, name)
        return self

    def initialize_database(
        self,
        stock_codes: Optional[Sequence[str]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        force: bool = False,
    ) -> "FreeMarketDataStore":
        """初始化本地 TSV 库；传入股票代码时同步预热日线数据。"""
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_field_description_file()
        if stock_codes:
            end_ts = as_timestamp(end_date) if end_date is not None else pd.Timestamp.today().normalize()
            start_ts = as_timestamp(start_date) if start_date is not None else end_ts - pd.DateOffset(years=self.history_years)
            self.get_daily(stock_codes, start_ts, end_ts, refresh=force)
        return self

    def add_field(
        self,
        field_name: str,
        description: str = "",
        table: str = "daily",
        dtype: str = "float64",
        source: str = "custom",
        default: object = np.nan,
        values: Optional[pd.DataFrame] = None,
    ) -> "FreeMarketDataStore":
        """为本地库添加字段，并自动更新字段说明文件。"""
        if table != "daily":
            raise NotImplementedError("当前版本仅支持 daily 表的字段扩展。")
        self._upsert_field_description(
            {"field": field_name, "table": table, "dtype": dtype, "description": description or "用户自定义字段", "source": source}
        )
        value_long = self._normalize_field_values(field_name, values) if values is not None else None
        for daily_file in self.daily_dir.glob("*.tsv"):
            frame = self._read_daily_file(daily_file)
            if frame.empty:
                continue
            if field_name not in frame.columns:
                frame[field_name] = default
            if value_long is not None:
                code = normalize_stock_code(frame["stock_code"].iloc[0])
                code_values = value_long[value_long["stock_code"] == code]
                if not code_values.empty:
                    frame = frame.drop(columns=[field_name], errors="ignore").merge(
                        code_values[["date", "stock_code", field_name]],
                        on=["date", "stock_code"],
                        how="left",
                    )
                    frame[field_name] = frame[field_name].fillna(default)
            self._write_daily_cache(frame["stock_code"].iloc[0], frame)
        return self

    def register_field_fetcher(
        self,
        field_name: str,
        fetcher: Callable[..., pd.DataFrame],
        description: str = "",
        dtype: str = "float64",
    ) -> "FreeMarketDataStore":
        """注册自定义字段拉取器，用于财报、北向资金等扩展字段。"""
        self._field_fetchers[field_name] = fetcher
        self._upsert_field_description(
            {
                "field": field_name,
                "table": "daily",
                "dtype": dtype,
                "description": description or "自定义字段拉取器生成字段",
                "source": "custom_fetcher",
            }
        )
        return self

    def describe_fields(self, table: Optional[str] = None) -> pd.DataFrame:
        """读取字段描述文件。"""
        self._ensure_field_description_file()
        fields = pd.read_csv(self.field_description_path, sep="\t")
        if table is not None:
            fields = fields[fields["table"] == table]
        return fields.reset_index(drop=True)

    def list_cached_symbols(self) -> list[str]:
        """列出当前已有日线缓存的股票代码。"""
        if not self.daily_dir.exists():
            return []
        return sorted(path.stem for path in self.daily_dir.glob("*.tsv"))

    def validate_database(self, stock_codes: Optional[Sequence[str]] = None, table: str = "daily") -> pd.DataFrame:
        """检查本地库字段、重复日期、价格逻辑和基础缺失情况。"""
        if table != "daily":
            raise NotImplementedError("当前版本仅支持 daily 表的数据验证。")
        codes = normalize_codes(stock_codes) if stock_codes else self.list_cached_symbols()
        issues: list[dict] = []
        required = ["date", "stock_code", "open", "high", "low", "close", "volume"]
        for code in codes:
            frame = self._read_daily_cache(code)
            if frame.empty:
                issues.append(self._issue(code, "error", "empty_cache", "本地没有该股票的日线缓存"))
                continue
            missing_columns = [col for col in required if col not in frame.columns]
            if missing_columns:
                issues.append(self._issue(code, "error", "missing_columns", ", ".join(missing_columns)))
                continue
            duplicates = frame.duplicated(["date", "stock_code"]).sum()
            if duplicates:
                issues.append(self._issue(code, "error", "duplicate_rows", f"重复行数: {duplicates}"))
            if not frame["date"].is_monotonic_increasing:
                issues.append(self._issue(code, "warning", "unsorted_dates", "日期未严格升序"))
            prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
            non_positive = (prices <= 0).any(axis=1).sum()
            if non_positive:
                issues.append(self._issue(code, "error", "non_positive_price", f"异常行数: {non_positive}"))
            bad_high = (prices["high"] < prices[["open", "low", "close"]].max(axis=1)).sum()
            bad_low = (prices["low"] > prices[["open", "high", "close"]].min(axis=1)).sum()
            if bad_high or bad_low:
                issues.append(self._issue(code, "error", "ohlc_inconsistent", f"high 异常: {bad_high}, low 异常: {bad_low}"))
            bad_volume = (pd.to_numeric(frame["volume"], errors="coerce") < 0).sum()
            if bad_volume:
                issues.append(self._issue(code, "error", "negative_volume", f"异常行数: {bad_volume}"))
            missing_close = prices["close"].isna().sum()
            if missing_close:
                issues.append(self._issue(code, "warning", "missing_close", f"缺失行数: {missing_close}"))
        return pd.DataFrame(issues, columns=["stock_code", "severity", "issue", "detail"])

    validate = validate_database

    def get_daily(
        self,
        stock_codes: Union[str, Sequence[str]],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        fields: Optional[Union[str, Sequence[str]]] = None,
        providers: Optional[Sequence[str]] = None,
        refresh: bool = False,
        return_format: ReturnFormat = "long",
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """获取日线数据。缓存覆盖时直接读 TSV，否则从免费数据源拉取并入库。"""
        codes = normalize_codes(stock_codes)
        requested_fields = self._normalize_fields(fields)
        start_ts = as_timestamp(start_date)
        end_ts = as_timestamp(end_date)
        if start_ts > end_ts:
            raise ValueError("start_date 不能晚于 end_date。")

        frames: list[pd.DataFrame] = []
        for code in codes:
            cached = self._read_daily_cache(code)
            if refresh or not self._cache_covers(cached, start_ts, end_ts, requested_fields):
                fetched = self._fetch_daily_from_providers(
                    code,
                    start_ts,
                    end_ts,
                    providers,
                    requested_fields,
                )
                self._merge_daily_cache(code, fetched)
                cached = self._read_daily_cache(code)
            cached = self._ensure_custom_fields(code, cached, start_ts, end_ts, requested_fields)
            frames.append(self._slice_daily(cached, start_ts, end_ts, requested_fields))

        result = self._empty_daily_frame() if not frames else pd.concat(frames, ignore_index=True).sort_values(["date", "stock_code"])
        self._ensure_field_descriptions_for_frame("daily", result)
        return self._format_daily_result(result, requested_fields, return_format)

    def extract_data(
        self,
        table: str = "daily",
        stock_codes: Optional[Union[str, Sequence[str]]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        fields: Optional[Union[str, Sequence[str]]] = None,
        return_format: ReturnFormat = "long",
        **kwargs,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """通用数据提取入口，目前 daily 表会走缓存和补库流程。"""
        if table != "daily":
            raise NotImplementedError("当前版本 extract_data 仅支持 daily 表。")
        if start_date is None or end_date is None:
            raise ValueError("提取 daily 数据必须提供 start_date 和 end_date。")
        codes = stock_codes or self.list_cached_symbols()
        if not codes:
            return self._format_daily_result(self._empty_daily_frame(), self._normalize_fields(fields), return_format)
        return self.get_daily(codes, start_date, end_date, fields=fields, return_format=return_format, **kwargs)

    def get_minute(
        self,
        stock_code: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "1m",
        provider: Optional[str] = None,
        adjust: str = "",
        fields: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """获取分钟级行情。该接口直连数据源，不写入本地 TSV。"""
        code = normalize_stock_code(stock_code)
        return self.provider.fetch_minute(
            code,
            start_date=start_date,
            end_date=end_date,
            period=period,
            adjust=adjust,
            providers=(provider,) if provider else None,
            fields=fields,
        )

    def get_realtime(
        self,
        stock_codes: Union[str, Sequence[str]],
        provider: Optional[str] = None,
        fields: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """获取实时行情。该接口直连数据源，不写入本地 TSV。"""
        codes = normalize_codes(stock_codes)
        return self.provider.fetch_realtime(codes, providers=(provider,) if provider else None, fields=fields)

    @staticmethod
    def to_matrices(frame: pd.DataFrame, fields: Sequence[str]) -> Dict[str, pd.DataFrame]:
        """将 long 格式日线表转换为回测常用的 T x N 矩阵字典。"""
        result = {}
        for field in fields:
            if field not in frame.columns:
                raise KeyError(f"字段不存在: {field}")
            result[field] = frame.pivot(index="date", columns="stock_code", values=field).sort_index()
        return result

    def _fetch_daily_from_providers(
        self,
        code: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        providers: Optional[Sequence[str]] = None,
        requested_fields: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        frame = self.provider.fetch_daily(
            code,
            start_date,
            end_date,
            providers=providers,
            fields=self._merge_provider_fields(self.provider.daily_fields, requested_fields),
        )
        return self._normalize_daily_frame(frame, code)

    @staticmethod
    def _merge_provider_fields(
        configured_fields: FieldSelection,
        requested_fields: Optional[Sequence[str]],
    ) -> FieldSelection:
        if not requested_fields or configured_fields is None:
            return configured_fields
        requested = list(requested_fields)
        if isinstance(configured_fields, dict):
            merged = {}
            for provider_name, provider_fields in configured_fields.items():
                merged[provider_name] = list(dict.fromkeys(FreeMarketDataStore._field_list(provider_fields) + requested))
            return merged
        return list(dict.fromkeys(FreeMarketDataStore._field_list(configured_fields) + requested))

    @staticmethod
    def _field_list(fields: Union[str, Sequence[str]]) -> list[str]:
        return [fields] if isinstance(fields, str) else list(fields)

    def _ensure_custom_fields(
        self,
        code: str,
        cached: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fields: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        if not fields:
            return cached
        for field in fields:
            if field in DAILY_INDEX_COLUMNS or field not in self._field_fetchers:
                continue
            has_values = False
            if field in cached.columns and not cached.empty:
                mask = (cached["date"] >= start_date.normalize()) & (cached["date"] <= end_date.normalize())
                has_values = not cached.loc[mask, field].isna().all()
            if has_values:
                continue
            fetcher = self._field_fetchers[field]
            try:
                values = fetcher(stock_code=code, start_date=start_date, end_date=end_date)
            except TypeError:
                values = fetcher(code, start_date, end_date)
            self.add_field(field, values=values)
            cached = self._read_daily_cache(code)
        return cached

    def _read_daily_cache(self, code: str) -> pd.DataFrame:
        return self._read_daily_file(self._daily_file(code))

    def _read_daily_file(self, path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return self._empty_daily_frame()
        frame = pd.read_csv(path, sep="\t")
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        if "stock_code" in frame.columns:
            frame["stock_code"] = frame["stock_code"].map(normalize_stock_code)
        return frame.sort_values(["date", "stock_code"]).reset_index(drop=True)

    def _merge_daily_cache(self, code: str, fetched: pd.DataFrame) -> None:
        old = self._read_daily_cache(code)
        combined = fetched if old.empty else pd.concat([old, fetched], ignore_index=True)
        self._write_daily_cache(code, combined)

    def _write_daily_cache(self, code: str, frame: pd.DataFrame) -> None:
        normalized = self._normalize_daily_frame(frame, normalize_stock_code(code))
        if normalized.empty:
            return
        normalized = normalized.drop_duplicates(["date", "stock_code"], keep="last")
        normalized = normalized.sort_values(["date", "stock_code"]).reset_index(drop=True)
        self._ensure_field_descriptions_for_frame("daily", normalized)
        output = normalized.copy()
        output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        output.to_csv(self._daily_file(code), sep="\t", index=False, encoding="utf-8")

    def _daily_file(self, code: str) -> Path:
        return self.daily_dir / f"{normalize_stock_code(code)}.tsv"

    def _cache_covers(
        self,
        frame: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fields: Optional[Sequence[str]],
    ) -> bool:
        if frame.empty or "date" not in frame.columns:
            return False
        required_fields = set(fields or [])
        required_fields.update(DAILY_PROVIDER_PACKAGE_COLUMNS)
        if required_fields and not required_fields.issubset(frame.columns):
            return False
        factor_window = frame.loc[(frame["date"] >= start_date.normalize()) & (frame["date"] <= end_date.normalize())]
        if factor_window.empty or factor_window[DAILY_PROVIDER_PACKAGE_COLUMNS].isna().any(axis=None):
            return False
        business_days = pd.bdate_range(start_date, end_date)
        start_check, end_check = (start_date.normalize(), end_date.normalize()) if business_days.empty else (business_days[0], business_days[-1])
        dates = pd.to_datetime(frame["date"])
        return dates.min() <= start_check and dates.max() >= end_check

    def _slice_daily(
        self,
        frame: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fields: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        sliced = frame.loc[(frame["date"] >= start_date.normalize()) & (frame["date"] <= end_date.normalize())].copy()
        if fields is None:
            return sliced
        columns = DAILY_INDEX_COLUMNS + [field for field in fields if field not in DAILY_INDEX_COLUMNS]
        missing = [field for field in columns if field not in sliced.columns]
        if missing:
            raise KeyError(f"本地数据缺少字段: {missing}")
        return sliced[columns]

    def _format_daily_result(
        self,
        frame: pd.DataFrame,
        fields: Optional[Sequence[str]],
        return_format: ReturnFormat,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if return_format == "long":
            return frame.reset_index(drop=True)
        value_fields = [col for col in frame.columns if col not in DAILY_INDEX_COLUMNS]
        if fields is not None:
            value_fields = [field for field in fields if field not in DAILY_INDEX_COLUMNS]
        if return_format == "wide":
            if len(value_fields) != 1:
                raise ValueError("return_format='wide' 需要且只能选择一个值字段。")
            return self.to_matrices(frame, value_fields)[value_fields[0]]
        if return_format == "dict":
            return self.to_matrices(frame, value_fields)
        raise ValueError(f"未知 return_format: {return_format}")

    def _ensure_field_description_file(self) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        if not self.field_description_path.exists():
            self._write_field_descriptions(pd.DataFrame(DEFAULT_FIELD_DESCRIPTIONS))
            return
        existing = self._read_field_descriptions()
        for field in DEFAULT_FIELD_DESCRIPTIONS:
            if not ((existing["table"] == field["table"]) & (existing["field"] == field["field"])).any():
                existing = pd.concat([existing, pd.DataFrame([field])], ignore_index=True)
        self._write_field_descriptions(existing)

    def _read_field_descriptions(self) -> pd.DataFrame:
        if not self.field_description_path.exists():
            return pd.DataFrame(columns=FIELD_DESCRIPTION_COLUMNS)
        return pd.read_csv(self.field_description_path, sep="\t")

    def _write_field_descriptions(self, fields: pd.DataFrame) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        ordered = fields[FIELD_DESCRIPTION_COLUMNS].drop_duplicates(["table", "field"], keep="last")
        ordered = ordered.sort_values(["table", "field"]).reset_index(drop=True)
        ordered.to_csv(self.field_description_path, sep="\t", index=False, encoding="utf-8")

    def _upsert_field_description(self, field: dict) -> None:
        self._ensure_field_description_file()
        fields = self._read_field_descriptions()
        mask = (fields["table"] == field["table"]) & (fields["field"] == field["field"])
        fields = pd.concat([fields.loc[~mask], pd.DataFrame([field])], ignore_index=True)
        self._write_field_descriptions(fields)

    def _ensure_field_descriptions_for_frame(self, table: str, frame: pd.DataFrame) -> None:
        self._ensure_field_description_file()
        if frame.empty:
            return
        fields = self._read_field_descriptions()
        existing = set(fields.loc[fields["table"] == table, "field"])
        additions = [
            {"field": column, "table": table, "dtype": str(frame[column].dtype), "description": "自动发现字段", "source": "auto"}
            for column in frame.columns
            if column not in existing
        ]
        if additions:
            self._write_field_descriptions(pd.concat([fields, pd.DataFrame(additions)], ignore_index=True))

    def _normalize_daily_frame(self, frame: pd.DataFrame, code: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return self._empty_daily_frame()
        normalized = frame.copy()
        if "datetime" in normalized.columns and "date" not in normalized.columns:
            normalized["date"] = normalized["datetime"]
        if "date" not in normalized.columns:
            raise ValueError("日线数据必须包含 date 字段。")
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
        normalized["stock_code"] = normalize_stock_code(code)
        dtype_map = self.describe_fields("daily").set_index("field")["dtype"].to_dict()
        for column, dtype in dtype_map.items():
            if column not in normalized.columns or column in {"date", "stock_code", "source", "updated_at"}:
                continue
            if str(dtype).startswith(("float", "int")):
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        return normalized.sort_values(["date", "stock_code"]).reset_index(drop=True)

    def _normalize_field_values(self, field_name: str, values: pd.DataFrame) -> pd.DataFrame:
        frame = values.copy()
        if {"date", "stock_code", field_name}.issubset(frame.columns):
            result = frame[["date", "stock_code", field_name]].copy()
        else:
            result = frame.stack(dropna=False).rename(field_name).reset_index()
            result.columns = ["date", "stock_code", field_name]
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
        result["stock_code"] = result["stock_code"].map(normalize_stock_code)
        return result

    @staticmethod
    def _normalize_fields(fields: Optional[Union[str, Iterable[str]]]) -> Optional[list[str]]:
        if fields is None:
            return None
        if isinstance(fields, str):
            return [fields]
        return list(fields)

    @staticmethod
    def _empty_daily_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=[field["field"] for field in DEFAULT_FIELD_DESCRIPTIONS if field["table"] == "daily"])

    @staticmethod
    def _issue(stock_code: str, severity: str, issue: str, detail: str) -> dict:
        return {"stock_code": stock_code, "severity": severity, "issue": issue, "detail": detail}

    normalize_stock_code = staticmethod(normalize_stock_code)