"""Clean TSV-backed database for HyperProvider daily data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Union

import pandas as pd

from .providers import DEFAULT_PROVIDER_CLASSES, BaseProvider, HyperProvider
from .symbols import as_timestamp, normalize_codes, normalize_stock_code

ReturnFormat = Literal["long", "wide", "dict"]
DAILY_INDEX_COLUMNS = ["date", "stock_code"]
DAILY_CORE_COLUMNS = ["open", "high", "low", "close", "volume", "qfq_factor", "hfq_factor", "price_source"]
MINUTE_INDEX_COLUMNS = ["datetime", "stock_code", "source"]
REALTIME_INDEX_COLUMNS = ["stock_code", "source", "timestamp"]
REALTIME_DEFAULT_FIELDS = ["price", "volume", "timestamp"]


class HyperDailyTsvDatabase:
    """Persist HyperProvider daily data to TSV and proxy minute/realtime requests."""

    DEFAULT_PROVIDERS = ("akshare", "baostock", "yahoo", "tencent", "xueqiu", "sina", "sohu")

    def __init__(
        self,
        db_path: Union[str, Path] = "data/hyper_daily_db",
        providers: Sequence[str] = DEFAULT_PROVIDERS,
        provider_instances: Optional[Sequence[BaseProvider]] = None,
        provider: Optional[HyperProvider] = None,
        daily_fields: Optional[Sequence[str]] = None,
        auto_initialize: bool = True,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.root_dir = self.db_path
        self.daily_dir = self.root_dir / "daily"
        self.providers = tuple(str(name).lower() for name in providers)

        provider_map: Dict[str, BaseProvider] = {name: cls() for name, cls in DEFAULT_PROVIDER_CLASSES.items()}
        self.hyper_provider = provider or HyperProvider(
            provider_map=provider_map,
            provider_order=self.providers,
            daily_fields=daily_fields,
        )
        self.provider = self.hyper_provider

        for instance in provider_instances or []:
            self.register_provider(instance)

        if auto_initialize:
            self.initialize_database()

    @classmethod
    def initialize(
        cls,
        root_dir: Union[str, Path] = "data/hyper_daily_db",
        hyper_provider: Optional[HyperProvider] = None,
        providers: Sequence[str] = DEFAULT_PROVIDERS,
        provider_instances: Optional[Sequence[BaseProvider]] = None,
        daily_fields: Optional[Sequence[str]] = None,
    ) -> "HyperDailyTsvDatabase":
        return cls(
            db_path=root_dir,
            providers=providers,
            provider_instances=provider_instances,
            provider=hyper_provider,
            daily_fields=daily_fields,
            auto_initialize=True,
        )

    def initialize_database(self) -> "HyperDailyTsvDatabase":
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        return self

    def register_provider(self, provider: BaseProvider, prefer_first: bool = False) -> "HyperDailyTsvDatabase":
        self.hyper_provider.register_provider(provider, prefer_first=prefer_first)
        name = str(getattr(provider, "name", "")).lower()
        if name and name not in self.providers:
            self.providers = (name, *self.providers) if prefer_first else (*self.providers, name)
        return self

    def list_cached_symbols(self) -> list[str]:
        if not self.daily_dir.exists():
            return []
        return sorted(path.stem for path in self.daily_dir.glob("*.tsv"))

    def describe_schema(self, stock_codes: Optional[Union[str, Sequence[str]]] = None) -> pd.DataFrame:
        codes = normalize_codes(stock_codes) if stock_codes is not None else self.list_cached_symbols()
        frames: list[pd.DataFrame] = []
        for code in codes:
            frame = self.read_daily(code)
            if not frame.empty:
                frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=["field", "dtype", "non_null_count", "sample_value"])

        merged = pd.concat(frames, ignore_index=True, sort=False)
        rows: list[dict[str, object]] = []
        for column in merged.columns:
            series = merged[column]
            sample_value = next((value for value in series if pd.notna(value)), None)
            rows.append(
                {
                    "field": column,
                    "dtype": str(series.dtype),
                    "non_null_count": int(series.notna().sum()),
                    "sample_value": sample_value,
                }
            )
        return pd.DataFrame(rows).sort_values(["field"]).reset_index(drop=True)

    def save_daily(
        self,
        stock_codes: Union[str, Sequence[str]],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        providers: Optional[Sequence[str]] = None,
        fields: Optional[Union[str, Sequence[str]]] = None,
        refresh: bool = True,
    ) -> pd.DataFrame:
        return self.get_daily(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            providers=providers,
            fields=fields,
            refresh=refresh,
            return_format="long",
        )

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
        codes = normalize_codes(stock_codes)
        requested_fields = self._normalize_fields(fields)
        start_ts = as_timestamp(start_date)
        end_ts = as_timestamp(end_date)
        if start_ts > end_ts:
            raise ValueError("start_date 不能晚于 end_date。")

        frames: list[pd.DataFrame] = []
        for code in codes:
            cached = self.read_daily(code)
            if refresh or not self._cache_covers(cached, start_ts, end_ts, requested_fields):
                fetched = self.hyper_provider.fetch_daily(
                    code,
                    start_ts,
                    end_ts,
                    providers=providers,
                    fields=requested_fields,
                )
                self.upsert_daily(fetched, stock_code=code)
                cached = self.read_daily(code)
            frames.append(self._slice_daily(cached, start_ts, end_ts, requested_fields))

        if not frames:
            result = self._empty_daily_frame()
        else:
            result = pd.concat(frames, ignore_index=True).sort_values(["date", "stock_code"]).reset_index(drop=True)
        return self._format_daily_result(result, requested_fields, return_format)

    def read_daily(self, stock_code: str) -> pd.DataFrame:
        path = self._daily_path(stock_code)
        if not path.exists() or path.stat().st_size == 0:
            return self._empty_daily_frame()
        return self._normalize_daily_frame(pd.read_csv(path, sep="\t"), normalize_stock_code(stock_code))

    def upsert_daily(self, frame: pd.DataFrame, stock_code: Optional[str] = None) -> "HyperDailyTsvDatabase":
        if frame is None or frame.empty:
            return self

        normalized = frame.copy()
        if stock_code is not None:
            normalized["stock_code"] = normalize_stock_code(stock_code)
        if "stock_code" not in normalized.columns:
            raise ValueError("写入日线数据时必须包含 stock_code 字段。")

        for code, code_frame in normalized.groupby("stock_code", sort=False):
            normalized_code = normalize_stock_code(code)
            existing = self.read_daily(normalized_code)
            prepared = self._normalize_daily_frame(code_frame, normalized_code)
            combined = prepared if existing.empty else pd.concat([existing, prepared], ignore_index=True)
            self._write_daily_file(normalized_code, combined)
        return self

    def get_minute(
        self,
        stock_codes: Union[str, Sequence[str]],
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "1m",
        adjust: str = "",
        providers: Optional[Sequence[str]] = None,
        fields: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        requested_fields = self._normalize_fields(fields)
        frames: list[pd.DataFrame] = []
        all_errors: Dict[str, Dict[str, str]] = {}

        for normalized_code in normalize_codes(stock_codes):
            errors: Dict[str, str] = {}
            for name in self._resolve_provider_names(providers):
                provider = self.hyper_provider.provider_map[name]
                try:
                    frame = provider.fetch_minute(
                        normalized_code,
                        start_date=start_date,
                        end_date=end_date,
                        period=period,
                        adjust=adjust,
                    )
                except Exception as exc:
                    errors[name] = str(exc)
                    continue
                if frame is None or frame.empty:
                    errors[name] = "返回空数据"
                    continue
                normalized = self._normalize_minute_frame(frame, normalized_code)
                frames.append(self._select_columns(normalized, MINUTE_INDEX_COLUMNS, requested_fields))
                break
            else:
                all_errors[normalized_code] = errors

        if all_errors:
            raise RuntimeError(f"无法获取分钟数据: {all_errors}")

        if not frames:
            return pd.DataFrame(columns=list(dict.fromkeys([*MINUTE_INDEX_COLUMNS, *(requested_fields or [])])))

        return pd.concat(frames, ignore_index=True).sort_values(["datetime", "stock_code"]).reset_index(drop=True)

    def get_realtime(
        self,
        stock_codes: Union[str, Sequence[str]],
        providers: Optional[Sequence[str]] = None,
        fields: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        requested_fields = self._normalize_fields(fields)
        output_fields = requested_fields if requested_fields is not None else REALTIME_DEFAULT_FIELDS
        result = self.hyper_provider.fetch_realtime(
            codes=normalize_codes(stock_codes),
            providers=providers,
            fields=output_fields,
        )
        return self._select_columns(result, ["stock_code", "timestamp"], output_fields)

    def validate_database(self, stock_codes: Optional[Sequence[str]] = None) -> pd.DataFrame:
        codes = normalize_codes(stock_codes) if stock_codes else self.list_cached_symbols()
        issues: list[dict] = []
        for code in codes:
            frame = self.read_daily(code)
            if frame.empty:
                issues.append({"stock_code": code, "severity": "error", "issue": "empty_cache", "detail": "本地没有该股票的日线缓存"})
                continue
            missing_columns = [column for column in DAILY_CORE_COLUMNS if column not in frame.columns]
            if missing_columns:
                issues.append({"stock_code": code, "severity": "error", "issue": "missing_columns", "detail": ", ".join(missing_columns)})
                continue
            duplicate_count = int(frame.duplicated(["date", "stock_code"]).sum())
            if duplicate_count:
                issues.append({"stock_code": code, "severity": "error", "issue": "duplicate_rows", "detail": f"重复行数: {duplicate_count}"})
            missing_core_rows = int(frame[DAILY_CORE_COLUMNS].isna().any(axis=1).sum())
            if missing_core_rows:
                issues.append({"stock_code": code, "severity": "error", "issue": "missing_core_values", "detail": f"异常行数: {missing_core_rows}"})
        return pd.DataFrame(issues, columns=["stock_code", "severity", "issue", "detail"])

    @staticmethod
    def to_matrices(frame: pd.DataFrame, fields: Sequence[str]) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for field in fields:
            if field not in frame.columns:
                raise KeyError(f"字段不存在: {field}")
            result[field] = frame.pivot(index="date", columns="stock_code", values=field).sort_index()
        return result

    @staticmethod
    def _normalize_fields(fields: Optional[Union[str, Sequence[str]]]) -> Optional[list[str]]:
        if fields is None:
            return None
        if isinstance(fields, str):
            return [fields]
        return list(dict.fromkeys(fields))

    @staticmethod
    def _empty_daily_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=DAILY_INDEX_COLUMNS)

    def _daily_path(self, stock_code: str) -> Path:
        return self.daily_dir / f"{normalize_stock_code(stock_code)}.tsv"

    def _resolve_provider_names(self, providers: Optional[Sequence[str]]) -> tuple[str, ...]:
        names = tuple(str(name).lower() for name in (providers or self.providers))
        missing = [name for name in names if name not in self.hyper_provider.provider_map]
        if missing:
            raise ValueError(f"未知 provider: {missing}")
        return names

    def _write_daily_file(self, stock_code: str, frame: pd.DataFrame) -> None:
        normalized = self._normalize_daily_frame(frame, stock_code)
        if normalized.empty:
            return
        output = normalized.copy()
        output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        output.to_csv(self._daily_path(stock_code), sep="\t", index=False, encoding="utf-8")

    def _normalize_daily_frame(self, frame: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return self._empty_daily_frame()
        normalized = frame.copy()
        if "datetime" in normalized.columns and "date" not in normalized.columns:
            normalized["date"] = normalized["datetime"]
        if "date" not in normalized.columns:
            raise ValueError("日线数据必须包含 date 字段。")
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
        normalized["stock_code"] = normalize_stock_code(stock_code)
        normalized = normalized.drop_duplicates(["date", "stock_code"], keep="last")
        return normalized.sort_values(["date", "stock_code"]).reset_index(drop=True)

    def _normalize_minute_frame(self, frame: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        normalized = frame.copy()
        if "datetime" not in normalized.columns and "date" in normalized.columns:
            normalized["datetime"] = normalized["date"]
        if "datetime" not in normalized.columns:
            raise ValueError("分钟数据必须包含 datetime 字段。")
        normalized["datetime"] = pd.to_datetime(normalized["datetime"])
        normalized["stock_code"] = normalize_stock_code(stock_code)
        if "source" not in normalized.columns:
            normalized["source"] = pd.NA
        return normalized.sort_values(["datetime", "stock_code"]).reset_index(drop=True)

    def _cache_covers(
        self,
        frame: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fields: Optional[Sequence[str]],
    ) -> bool:
        if frame.empty or "date" not in frame.columns:
            return False
        required_fields = set(DAILY_CORE_COLUMNS)
        required_fields.update(fields or [])
        if not required_fields.issubset(frame.columns):
            return False
        window = frame.loc[(frame["date"] >= start_date.normalize()) & (frame["date"] <= end_date.normalize())]
        if window.empty:
            return False
        if window[list(required_fields)].isna().any(axis=None):
            return False
        business_days = pd.bdate_range(start_date, end_date)
        if business_days.empty:
            return True
        return window["date"].min() <= business_days[0] and window["date"].max() >= business_days[-1]

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
        return self._select_columns(sliced, DAILY_INDEX_COLUMNS, fields)

    def _select_columns(
        self,
        frame: pd.DataFrame,
        protected_columns: Sequence[str],
        fields: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        if fields is None:
            return frame.reset_index(drop=True)
        columns = [column for column in [*protected_columns, *fields] if column in frame.columns]
        missing = [column for column in fields if column not in frame.columns]
        if missing:
            raise KeyError(f"数据缺少字段: {missing}")
        return frame.loc[:, list(dict.fromkeys(columns))].reset_index(drop=True)

    def _format_daily_result(
        self,
        frame: pd.DataFrame,
        fields: Optional[Sequence[str]],
        return_format: ReturnFormat,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if return_format == "long":
            return frame.reset_index(drop=True)
        value_fields = [column for column in frame.columns if column not in DAILY_INDEX_COLUMNS]
        if fields is not None:
            value_fields = [field for field in fields if field not in DAILY_INDEX_COLUMNS]
        if return_format == "wide":
            if len(value_fields) != 1:
                raise ValueError("return_format='wide' 需要且只能选择一个值字段。")
            return self.to_matrices(frame, value_fields)[value_fields[0]]
        if return_format == "dict":
            return self.to_matrices(frame, value_fields)
        raise ValueError(f"未知 return_format: {return_format}")
