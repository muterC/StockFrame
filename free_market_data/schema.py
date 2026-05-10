"""Schema constants for the standalone free market data store."""

from __future__ import annotations

FIELD_DESCRIPTION_COLUMNS = ["field", "table", "dtype", "description", "source"]
DAILY_INDEX_COLUMNS = ["date", "stock_code"]

DEFAULT_FIELD_DESCRIPTIONS = [
    {"field": "date", "table": "daily", "dtype": "datetime64[ns]", "description": "交易日期", "source": "system"},
    {"field": "stock_code", "table": "daily", "dtype": "object", "description": "标准化股票代码，如 600000.SH 或 000001.SZ", "source": "system"},
    {"field": "open", "table": "daily", "dtype": "float64", "description": "未复权开盘价", "source": "system"},
    {"field": "high", "table": "daily", "dtype": "float64", "description": "未复权最高价", "source": "system"},
    {"field": "low", "table": "daily", "dtype": "float64", "description": "未复权最低价", "source": "system"},
    {"field": "close", "table": "daily", "dtype": "float64", "description": "未复权收盘价", "source": "system"},
    {"field": "volume", "table": "daily", "dtype": "float64", "description": "成交量", "source": "system"},
    {"field": "amount", "table": "daily", "dtype": "float64", "description": "成交额", "source": "system"},
    {"field": "amplitude", "table": "daily", "dtype": "float64", "description": "振幅，通常为百分比数值", "source": "system"},
    {"field": "pct_change", "table": "daily", "dtype": "float64", "description": "涨跌幅，通常为百分比数值", "source": "system"},
    {"field": "change", "table": "daily", "dtype": "float64", "description": "涨跌额", "source": "system"},
    {"field": "turnover", "table": "daily", "dtype": "float64", "description": "换手率，通常为百分比数值", "source": "system"},
    {"field": "pre_close", "table": "daily", "dtype": "float64", "description": "前收盘价", "source": "system"},
    {"field": "adj_close", "table": "daily", "dtype": "float64", "description": "复权收盘价，主要来自 Yahoo", "source": "system"},
    {"field": "qfq_factor", "table": "daily", "dtype": "float64", "description": "前复权因子，前复权价格 = 未复权价格 × qfq_factor", "source": "system"},
    {"field": "hfq_factor", "table": "daily", "dtype": "float64", "description": "后复权因子，后复权价格 = 未复权价格 × hfq_factor", "source": "system"},
    {"field": "price_source", "table": "daily", "dtype": "object", "description": "未复权 OHLC 与 qfq_factor/hfq_factor 的同源 provider", "source": "system"},
    {"field": "pe_ttm", "table": "daily", "dtype": "float64", "description": "滚动市盈率", "source": "system"},
    {"field": "pb", "table": "daily", "dtype": "float64", "description": "市净率", "source": "system"},
    {"field": "ps_ttm", "table": "daily", "dtype": "float64", "description": "滚动市销率", "source": "system"},
    {"field": "is_st", "table": "daily", "dtype": "float64", "description": "是否 ST，通常 1 表示 ST，0 表示非 ST", "source": "system"},
    {"field": "total_market_cap", "table": "daily", "dtype": "float64", "description": "总市值，可由数据源或自定义字段补齐", "source": "system"},
    {"field": "float_market_cap", "table": "daily", "dtype": "float64", "description": "流通市值，可由数据源或自定义字段补齐", "source": "system"},
    {"field": "northbound_net_buy", "table": "daily", "dtype": "float64", "description": "北向资金净买入额，可由自定义字段补齐", "source": "system"},
    {"field": "revenue", "table": "daily", "dtype": "float64", "description": "营业收入，可由财报字段补齐", "source": "system"},
    {"field": "net_profit", "table": "daily", "dtype": "float64", "description": "归母净利润，可由财报字段补齐", "source": "system"},
    {"field": "source", "table": "daily", "dtype": "object", "description": "该行数据来源", "source": "system"},
    {"field": "updated_at", "table": "daily", "dtype": "datetime64[ns]", "description": "本地缓存更新时间", "source": "system"},
]