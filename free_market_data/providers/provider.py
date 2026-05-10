"""Unified provider facade used by the local TSV database."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import pandas as pd

from .base import BaseProvider, FieldSelection, PROTECTED_COLUMNS

REQUIRED_DAILY_PRICE_FACTOR_COLUMNS = ("open", "high", "low", "close", "qfq_factor", "hfq_factor")
DAILY_QUOTE_PACKAGE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pre_close",
    "change",
    "pct_change",
    "amplitude",
    "turnover",
    "adj_close",
    "qfq_factor",
    "hfq_factor",
    "price_source",
)


class Provider:
    """
    Final provider facade for database access.

    It owns the provider instances, merges daily data across all configured providers,
    applies fallback order for direct minute/realtime endpoints, annotates source columns,
    and optionally keeps only selected fields per endpoint or per provider.
    """

    def __init__(
        self,
        provider_map: Dict[str, BaseProvider],
        provider_order: Sequence[str],
        daily_fields: FieldSelection = None,
        minute_fields: FieldSelection = None,
        realtime_fields: FieldSelection = None,
    ) -> None:
        self.provider_map = provider_map
        self.provider_order = tuple(str(provider).lower() for provider in provider_order)
        self.daily_fields = daily_fields
        self.minute_fields = minute_fields
        self.realtime_fields = realtime_fields

    def register_provider(self, provider: BaseProvider, prefer_first: bool = False) -> None:
        name = str(getattr(provider, "name", "")).lower()
        if not name:
            raise ValueError("provider 必须提供 name 属性。")
        self.provider_map[name] = provider
        if name not in self.provider_order:
            self.provider_order = (name, *self.provider_order) if prefer_first else (*self.provider_order, name)

    def fetch_daily(
        self,
        code: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        providers: Optional[Sequence[str]] = None,
        fields: FieldSelection = None,
    ) -> pd.DataFrame:
        errors: Dict[str, str] = {}
        frames: list[pd.DataFrame] = []
        field_policy = fields if fields is not None else self.daily_fields
        for provider in self._provider_candidates(providers):
            try:
                frame = self._fetch_daily_with_adjustment_factors(provider, code, start_date, end_date)
                if frame.empty:
                    errors[provider.name] = "返回空数据"
                    continue
                frame = frame.copy()
                frame["source"] = provider.name
                frame["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                frames.append(self._select_fields(frame, "daily", provider.name, field_policy))
            except Exception as exc:
                errors[provider.name] = str(exc)
        if not frames:
            raise RuntimeError(f"无法获取 {code} 的日线数据: {errors}")
        return self._merge_daily_frames(frames)

    def fetch_minute(
        self,
        code: str,
        start_date: object = None,
        end_date: object = None,
        period: str = "1m",
        adjust: str = "",
        providers: Optional[Sequence[str]] = None,
        fields: FieldSelection = None,
    ) -> pd.DataFrame:
        errors: Dict[str, str] = {}
        for provider in self._provider_candidates(providers):
            try:
                frame = provider.fetch_minute(code, start_date, end_date, period, adjust)
                if frame.empty:
                    errors[provider.name] = "返回空数据"
                    continue
                frame = frame.copy()
                if "source" not in frame.columns:
                    frame["source"] = provider.name
                return self._select_fields(frame, "minute", provider.name, fields if fields is not None else self.minute_fields)
            except Exception as exc:
                errors[provider.name] = str(exc)
        raise RuntimeError(f"无法获取 {code} 的分钟级数据: {errors}")

    def fetch_realtime(
        self,
        codes: Sequence[str],
        providers: Optional[Sequence[str]] = None,
        fields: FieldSelection = None,
    ) -> pd.DataFrame:
        errors: Dict[str, str] = {}
        for provider in self._provider_candidates(providers):
            try:
                frame = provider.fetch_realtime(codes)
                if frame.empty:
                    errors[provider.name] = "返回空数据"
                    continue
                frame = frame.copy()
                if "source" not in frame.columns:
                    frame["source"] = provider.name
                if "timestamp" not in frame.columns:
                    frame["timestamp"] = pd.Timestamp.now()
                return self._select_fields(frame, "realtime", provider.name, fields if fields is not None else self.realtime_fields)
            except Exception as exc:
                errors[provider.name] = str(exc)
        raise RuntimeError(f"无法获取实时行情: {errors}")

    def _provider_candidates(self, providers: Optional[Sequence[str]] = None) -> list[BaseProvider]:
        names = tuple(str(provider).lower() for provider in (providers or self.provider_order))
        result = []
        for name in names:
            provider = self.provider_map.get(name)
            if provider is None:
                raise ValueError(f"未知 provider: {name}")
            result.append(provider)
        return result

    def _fetch_daily_with_adjustment_factors(
        self,
        provider: BaseProvider,
        code: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        raw = self._fetch_daily_variant(provider, code, start_date, end_date, "")
        if raw.empty:
            return raw

        raw = self._attach_factor_from_adj_close(raw)
        adjustments = set(str(item or "") for item in getattr(provider, "daily_adjustments", ("",)))
        factor_errors: Dict[str, str] = {}
        for adjustment, factor_column in (("qfq", "qfq_factor"), ("hfq", "hfq_factor")):
            if adjustment not in adjustments:
                factor_errors[factor_column] = f"{provider.name} 未声明支持 {adjustment} 日线"
                continue
            try:
                adjusted = self._fetch_daily_variant(provider, code, start_date, end_date, adjustment)
            except Exception as exc:
                factor_errors[factor_column] = str(exc)
                continue
            raw = self._attach_adjustment_factor(raw, adjusted, factor_column)
        return self._require_daily_price_factor_package(raw, provider.name, factor_errors)

    def _fetch_daily_variant(
        self,
        provider: BaseProvider,
        code: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        adjustment: str,
    ) -> pd.DataFrame:
        frame = provider.fetch_daily_adjusted(code, start_date, end_date, adjustment)
        return self._prepare_daily_frame(frame, code)

    @staticmethod
    def _prepare_daily_frame(frame: pd.DataFrame, code: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        prepared = frame.copy()
        if "datetime" in prepared.columns and "date" not in prepared.columns:
            prepared["date"] = prepared["datetime"]
        if "date" not in prepared.columns:
            raise ValueError("日线数据必须包含 date 字段。")
        prepared["date"] = pd.to_datetime(prepared["date"]).dt.normalize()
        prepared["stock_code"] = code
        return prepared.drop_duplicates(["date", "stock_code"], keep="last").sort_values(["date", "stock_code"]).reset_index(drop=True)

    @staticmethod
    def _attach_factor_from_adj_close(frame: pd.DataFrame) -> pd.DataFrame:
        if "adj_close" not in frame.columns or "close" not in frame.columns:
            return frame
        result = frame.copy()
        raw_close = pd.to_numeric(result["close"], errors="coerce")
        adjusted_close = pd.to_numeric(result["adj_close"], errors="coerce")
        factor = adjusted_close / raw_close
        factor = factor.where(raw_close.notna() & adjusted_close.notna() & raw_close.ne(0))
        factor = factor.where(factor.abs() != float("inf"))
        if "qfq_factor" in result.columns:
            result["qfq_factor"] = pd.to_numeric(result["qfq_factor"], errors="coerce").combine_first(factor)
        else:
            result["qfq_factor"] = factor
        return result

    @staticmethod
    def _attach_adjustment_factor(raw: pd.DataFrame, adjusted: pd.DataFrame, factor_column: str) -> pd.DataFrame:
        if raw.empty or adjusted.empty or "close" not in raw.columns or "close" not in adjusted.columns:
            return raw
        raw_indexed = raw.set_index(["date", "stock_code"]).copy()
        adjusted_close = pd.to_numeric(
            adjusted.set_index(["date", "stock_code"]).reindex(raw_indexed.index)["close"],
            errors="coerce",
        )
        raw_close = pd.to_numeric(raw_indexed["close"], errors="coerce")
        factor = adjusted_close / raw_close
        factor = factor.where(raw_close.notna() & adjusted_close.notna() & raw_close.ne(0))
        factor = factor.where(factor.abs() != float("inf"))
        if factor_column in raw_indexed.columns:
            raw_indexed[factor_column] = pd.to_numeric(raw_indexed[factor_column], errors="coerce").combine_first(factor)
        else:
            raw_indexed[factor_column] = factor
        return raw_indexed.reset_index()

    @staticmethod
    def _require_daily_price_factor_package(frame: pd.DataFrame, provider_name: str, factor_errors: Dict[str, str]) -> pd.DataFrame:
        missing_columns = [column for column in REQUIRED_DAILY_PRICE_FACTOR_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise RuntimeError(
                f"{provider_name} 日线数据缺少同源行情/复权因子字段: {missing_columns}; "
                f"复权因子必须由该 provider 的未复权 close 与复权 close 同步计算。详情: {factor_errors}"
            )
        result = frame.copy()
        for column in REQUIRED_DAILY_PRICE_FACTOR_COLUMNS:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        invalid_mask = result[list(REQUIRED_DAILY_PRICE_FACTOR_COLUMNS)].isna().any(axis=1)
        invalid_mask |= result[["open", "high", "low", "close", "qfq_factor", "hfq_factor"]].le(0).any(axis=1)
        if invalid_mask.all():
            raise RuntimeError(
                f"{provider_name} 日线数据没有任何一行同时具备未复权 OHLC、qfq_factor、hfq_factor。详情: {factor_errors}"
            )
        result = result.loc[~invalid_mask].copy()
        result["price_source"] = provider_name
        return result

    @staticmethod
    def _merge_daily_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
        combined: Optional[pd.DataFrame] = None
        source_by_index: dict[tuple[pd.Timestamp, str], list[str]] = {}
        for frame in frames:
            if frame.empty:
                continue
            prepared = frame.copy()
            prepared["date"] = pd.to_datetime(prepared["date"]).dt.normalize()
            prepared = prepared.drop_duplicates(["date", "stock_code"], keep="last")
            indexed = prepared.set_index(["date", "stock_code"])

            source_series = indexed["source"] if "source" in indexed.columns else pd.Series("", index=indexed.index)

            values = indexed.drop(columns=["source", "updated_at"], errors="ignore")
            if combined is None:
                combined = values
                Provider._append_used_sources(source_by_index, source_series, values.index)
                continue

            existing_index = combined.index
            merged_index = existing_index.union(values.index)
            new_index = values.index.difference(existing_index)
            combined = combined.reindex(merged_index)
            values = values.reindex(merged_index)
            used_index = values.index[:0]
            for column in values.columns:
                if column not in combined.columns:
                    combined[column] = pd.NA
                if column in DAILY_QUOTE_PACKAGE_COLUMNS:
                    update_index = new_index.intersection(values.index[values[column].notna()])
                else:
                    update_mask = combined[column].isna() & values[column].notna()
                    update_index = values.index[update_mask]
                if len(update_index):
                    combined.loc[update_index, column] = values.loc[update_index, column]
                    used_index = used_index.union(update_index)
            Provider._append_used_sources(source_by_index, source_series.reindex(merged_index), used_index)

        if combined is None or combined.empty:
            return pd.DataFrame()
        result = combined.reset_index()
        result["source"] = [",".join(source_by_index.get((row.date, row.stock_code), [])) for row in result.itertuples(index=False)]
        result["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        return result.sort_values(["date", "stock_code"]).reset_index(drop=True)

    @staticmethod
    def _append_used_sources(
        source_by_index: dict[tuple[pd.Timestamp, str], list[str]],
        source_series: pd.Series,
        used_index: pd.Index,
    ) -> None:
        for index_value in used_index:
            sources = source_by_index.setdefault(index_value, [])
            for item in str(source_series.get(index_value, "")).split(","):
                item = item.strip()
                if item and item not in sources:
                    sources.append(item)

    def _select_fields(
        self,
        frame: pd.DataFrame,
        endpoint: str,
        provider_name: str,
        fields: FieldSelection,
    ) -> pd.DataFrame:
        selected = self._resolve_field_selection(fields, provider_name)
        if selected is None:
            return frame
        protected = list(PROTECTED_COLUMNS[endpoint])
        columns = [column for column in protected + selected if column in frame.columns]
        return frame.loc[:, list(dict.fromkeys(columns))]

    @staticmethod
    def _resolve_field_selection(fields: FieldSelection, provider_name: str) -> Optional[list[str]]:
        if fields is None:
            return None
        if isinstance(fields, dict):
            selected = fields.get(provider_name, fields.get("default"))
            if selected is None:
                return None
            return [selected] if isinstance(selected, str) else list(selected)
        return [fields] if isinstance(fields, str) else list(fields)