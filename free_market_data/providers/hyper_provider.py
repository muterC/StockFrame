"""Hyper provider that assembles one core daily package plus optional enrichments."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import pandas as pd

from .base import BaseProvider, FieldSelection, PROTECTED_COLUMNS

PRICE_CORE_COLUMNS = ("open", "high", "low", "close", "volume", "qfq_factor", "hfq_factor")
PRICE_CORE_PACKAGE_COLUMNS = PRICE_CORE_COLUMNS + ("price_source",)


class HyperProvider:
    """Provider facade for daily assembly and lightweight realtime fallback."""

    def __init__(
        self,
        provider_map: Dict[str, BaseProvider],
        provider_order: Sequence[str],
        daily_fields: FieldSelection = None,
    ) -> None:
        self.provider_map = provider_map
        self.provider_order = tuple(str(provider).lower() for provider in provider_order)
        self.daily_fields = daily_fields

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
        provider_names = tuple(str(provider).lower() for provider in (providers or self.provider_order))
        provider_list: list[BaseProvider] = []
        for name in provider_names:
            provider = self.provider_map.get(name)
            if provider is None:
                raise ValueError(f"未知 provider: {name}")
            provider_list.append(provider)

        def prepare_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
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

        def attach_adjustment_factor(raw_frame: pd.DataFrame, adjusted_frame: pd.DataFrame, factor_column: str) -> pd.DataFrame:
            if raw_frame.empty or adjusted_frame.empty or "close" not in raw_frame.columns or "close" not in adjusted_frame.columns:
                return raw_frame
            raw_indexed = raw_frame.set_index(["date", "stock_code"]).copy()
            adjusted_close = pd.to_numeric(
                adjusted_frame.set_index(["date", "stock_code"]).reindex(raw_indexed.index)["close"],
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

        errors: Dict[str, str] = {}
        core_frame: Optional[pd.DataFrame] = None
        core_provider_name: Optional[str] = None
        enrichment_frames: list[tuple[str, pd.DataFrame]] = []

        for provider in provider_list:
            try:
                raw_frame = prepare_daily_frame(provider.fetch_daily_adjusted(code, start_date, end_date, ""))
            except Exception as exc:
                errors[provider.name] = f"raw 日线失败: {exc}"
                continue

            if raw_frame.empty:
                errors[provider.name] = "返回空数据"
                continue

            if core_frame is None:
                candidate = raw_frame.copy()
                if "adj_close" in candidate.columns and "close" in candidate.columns:
                    raw_close = pd.to_numeric(candidate["close"], errors="coerce")
                    adjusted_close = pd.to_numeric(candidate["adj_close"], errors="coerce")
                    qfq_factor = adjusted_close / raw_close
                    qfq_factor = qfq_factor.where(raw_close.notna() & adjusted_close.notna() & raw_close.ne(0))
                    qfq_factor = qfq_factor.where(qfq_factor.abs() != float("inf"))
                    if "qfq_factor" in candidate.columns:
                        candidate["qfq_factor"] = pd.to_numeric(candidate["qfq_factor"], errors="coerce").combine_first(qfq_factor)
                    else:
                        candidate["qfq_factor"] = qfq_factor

                adjustments = set(str(item or "") for item in getattr(provider, "daily_adjustments", ("",)))
                factor_errors: Dict[str, str] = {}
                for adjustment, factor_column in (("qfq", "qfq_factor"), ("hfq", "hfq_factor")):
                    if adjustment not in adjustments:
                        factor_errors[factor_column] = f"{provider.name} 未声明支持 {adjustment} 日线"
                        continue
                    try:
                        adjusted_frame = prepare_daily_frame(provider.fetch_daily_adjusted(code, start_date, end_date, adjustment))
                    except Exception as exc:
                        factor_errors[factor_column] = str(exc)
                        continue
                    candidate = attach_adjustment_factor(candidate, adjusted_frame, factor_column)

                missing_columns = [column for column in PRICE_CORE_COLUMNS if column not in candidate.columns]
                if missing_columns:
                    errors[provider.name] = f"{provider.name} 日线数据缺少核心字段: {missing_columns}; 详情: {factor_errors}"
                else:
                    for column in PRICE_CORE_COLUMNS:
                        candidate[column] = pd.to_numeric(candidate[column], errors="coerce")
                    invalid_mask = candidate[list(PRICE_CORE_COLUMNS)].isna().any(axis=1)
                    invalid_mask |= candidate[list(PRICE_CORE_COLUMNS)].le(0).any(axis=1)
                    candidate = candidate.loc[~invalid_mask].copy()
                    if candidate.empty:
                        errors[provider.name] = f"{provider.name} 日线数据没有任何一行同时具备核心行情与复权因子。详情: {factor_errors}"
                    else:
                        candidate["price_source"] = provider.name
                        candidate["source"] = provider.name
                        candidate["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        core_frame = candidate.sort_values(["date", "stock_code"]).reset_index(drop=True)
                        core_provider_name = provider.name
                        continue

            enrichment_frames.append((provider.name, raw_frame))

        if core_frame is None or core_provider_name is None:
            raise RuntimeError(f"无法获取 {code} 的日线核心包（OHLC + qfq_factor + hfq_factor）: {errors}")

        merged = core_frame.copy().set_index(["date", "stock_code"])
        for provider_name, frame in enrichment_frames:
            enrichment = frame.copy().set_index(["date", "stock_code"]).reindex(merged.index)
            enrichment = enrichment.dropna(how="all")
            if enrichment.empty:
                continue
            used_index = enrichment.index[:0]
            for column in enrichment.columns:
                if column in {"source", "updated_at", *PRICE_CORE_PACKAGE_COLUMNS}:
                    continue
                if column not in merged.columns:
                    merged[column] = pd.NA
                update_mask = merged[column].isna() & enrichment[column].notna()
                update_index = enrichment.index[update_mask]
                if len(update_index):
                    merged.loc[update_index, column] = enrichment.loc[update_index, column]
                    used_index = used_index.union(update_index)
            if len(used_index):
                source_series = merged["source"].astype("object")
                for index_value in used_index:
                    current = [item.strip() for item in str(source_series.get(index_value, "")).split(",") if item.strip()]
                    if provider_name not in current:
                        current.append(provider_name)
                    source_series.loc[index_value] = ",".join(current)
                merged["source"] = source_series

        result = merged.reset_index()
        result["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        selected_fields = fields if fields is not None else self.daily_fields
        if selected_fields is None:
            final_result = result.sort_values(["date", "stock_code"]).reset_index(drop=True)
        elif isinstance(selected_fields, dict):
            selected = selected_fields.get(core_provider_name, selected_fields.get("default"))
            if selected is None:
                final_result = result.sort_values(["date", "stock_code"]).reset_index(drop=True)
            else:
                selected = [selected] if isinstance(selected, str) else list(selected)
                protected = list(PROTECTED_COLUMNS["daily"])
                columns = [column for column in protected + selected if column in result.columns]
                final_result = result.loc[:, list(dict.fromkeys(columns))].sort_values(["date", "stock_code"]).reset_index(drop=True)
        else:
            selected = [selected_fields] if isinstance(selected_fields, str) else list(selected_fields)
            protected = list(PROTECTED_COLUMNS["daily"])
            columns = [column for column in protected + selected if column in result.columns]
            final_result = result.loc[:, list(dict.fromkeys(columns))].sort_values(["date", "stock_code"]).reset_index(drop=True)

        return final_result

    def fetch_realtime(
        self,
        codes: Sequence[str],
        providers: Optional[Sequence[str]] = None,
        fields: FieldSelection = None,
    ) -> pd.DataFrame:
        provider_names = tuple(str(provider).lower() for provider in (providers or self.provider_order))
        provider_list: list[BaseProvider] = []
        for name in provider_names:
            provider = self.provider_map.get(name)
            if provider is None:
                raise ValueError(f"未知 provider: {name}")
            provider_list.append(provider)

        errors: Dict[str, str] = {}
        normalized_codes = tuple(str(code) for code in codes)
        for provider in provider_list:
            try:
                frame = provider.fetch_realtime(normalized_codes)
            except Exception as exc:
                errors[provider.name] = str(exc)
                continue

            if frame is None or frame.empty:
                errors[provider.name] = "返回空数据"
                continue

            result = frame.copy()
            if "stock_code" not in result.columns:
                errors[provider.name] = "缺少 stock_code 字段"
                continue
            result = result[result["stock_code"].isin(normalized_codes)].copy()
            if result.empty:
                errors[provider.name] = "未返回请求的股票代码"
                continue

            if "price" in result.columns:
                result["price"] = pd.to_numeric(result["price"], errors="coerce")
            if "volume" in result.columns:
                result["volume"] = pd.to_numeric(result["volume"], errors="coerce")
            if "amount" in result.columns:
                result["amount"] = pd.to_numeric(result["amount"], errors="coerce")
            if "timestamp" not in result.columns:
                result["timestamp"] = pd.Timestamp.now()
            else:
                result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce").fillna(pd.Timestamp.now())
            if "source" not in result.columns:
                result["source"] = provider.name

            default_columns = [
                "stock_code",
                "name",
                "price",
                "open",
                "high",
                "low",
                "pre_close",
                "change",
                "pct_change",
                "volume",
                "amount",
                "timestamp",
                "source",
            ]
            selected_fields = fields if fields is not None else default_columns
            if isinstance(selected_fields, dict):
                selected = selected_fields.get(provider.name, selected_fields.get("default", default_columns))
                selected = [selected] if isinstance(selected, str) else list(selected)
            else:
                selected = [selected_fields] if isinstance(selected_fields, str) else list(selected_fields)
            columns = [column for column in ["stock_code", *selected] if column in result.columns]
            missing_codes = sorted(set(normalized_codes) - set(result["stock_code"].dropna().astype(str)))
            if missing_codes:
                errors[provider.name] = f"缺少股票代码: {missing_codes}"
                continue

            required_columns = [column for column in selected if column not in {"name", "source"}]
            missing_columns = [column for column in required_columns if column not in result.columns]
            if missing_columns:
                errors[provider.name] = f"缺少字段: {missing_columns}"
                continue

            invalid_columns = [
                column
                for column in required_columns
                if column != "timestamp" and result[column].isna().any()
            ]
            if invalid_columns:
                errors[provider.name] = f"字段存在空值: {invalid_columns}"
                continue

            return result.loc[:, list(dict.fromkeys(columns))].reset_index(drop=True)

        raise RuntimeError(f"无法获取实时行情: {errors}")