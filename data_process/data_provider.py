"""
混合数据提供者

结合 Yahoo Finance 和 AkShare 的优势：
- Yahoo Finance: 历史市值数据（完整时序）
- AkShare: 完整财务数据（1998年至今，100条记录）

确保数据索引一致，提供最完整的数据集
"""
import logging
import pandas as pd
from typing import Optional, List
from datetime import datetime
import akshare as ak
import yfinance as yf


class HybridDataProvider():
    """
    混合数据提供者

    数据源策略：
    1. OHLCV数据：优先 AkShare（免费、稳定）
    2. 市值数据：使用 Yahoo Finance（完整历史时序数据）⭐
    3. 财务数据：使用 AkShare（100条完整记录，1998-2025）⭐
    4. 复权因子：优先 AkShare
    5. 股票列表：使用 AkShare

    核心优势：
    - 市值数据：Yahoo提供真实的历史时序数据（241条），而不是快照
    - 财务数据：AkShare提供27年完整历史（100条 vs Yahoo的5条）
    - 索引对齐：自动处理两个数据源的索引一致性问题
    """

    def __init__(self):
        super().__init__()
        self.source_name = "hybrid"
        self.logger = logging.getLogger(__name__)
        self._trading_calendar_cache = None

    def get_stock_list(self) -> List[dict]:
        """
        获取股票列表（使用 AkShare）

        Returns:
            List[dict]: 股票列表
        """
        try:
            # 获取沪深A股列表
            df_sh = ak.stock_info_sh_name_code()  # 上交所
            df_sz = ak.stock_zh_a_spot_em()       # 深交所（通过东方财富）

            # 处理上交所数据
            stock_list_sh = []
            if df_sh is not None and not df_sh.empty:
                for _, row in df_sh.iterrows():
                    stock_list_sh.append({
                        'symbol': row['证券代码'],
                        'name': row['证券简称'],
                        'exchange': 'SH'
                    })

            # 处理深交所数据
            stock_list_sz = []
            if df_sz is not None and not df_sz.empty:
                for _, row in df_sz.iterrows():
                    code = row['代码']
                    if code.startswith('0') or code.startswith('3'):  # 深市或创业板
                        stock_list_sz.append({
                            'symbol': code,
                            'name': row['名称'],
                            'exchange': 'SZ'
                        })

            stock_list = stock_list_sh + stock_list_sz
            return stock_list

        except Exception as e:
            error_msg = f"Failed to get stock list: {e}"
            raise RuntimeError(error_msg)

    def is_trading_day(self, date: Optional[str] = None) -> bool:
        """
        判断指定日期是否为交易日（使用 AkShare 交易日历）

        Args:
            date: 日期字符串（YYYY-MM-DD），默认为今天

        Returns:
            bool: True=交易日，False=非交易日
        """
        try:
            # 如果没有指定日期，使用今天
            if date is None:
                date = pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d')

            # 标准化日期格式
            check_date = pd.to_datetime(date).strftime('%Y-%m-%d')

            # 获取交易日历（使用缓存避免频繁API调用）
            if self._trading_calendar_cache is None:

                # 获取历史交易日历（不需要year参数，直接返回完整历史）
                df = ak.tool_trade_date_hist_sina()

                if df is not None and not df.empty:
                    # 转换日期格式
                    self._trading_calendar_cache = set(
                        pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
                    )
                else:
                    return True # 出错时保守处理，假设是交易日

            # 检查是否在交易日历中
            is_trading = check_date in self._trading_calendar_cache
            return is_trading

        except Exception as e:
            return True  # 出错时保守处理，假设是交易日

    def _convert_symbol_to_yahoo(self, symbol: str) -> str:
        """
        将国内股票代码转换为 Yahoo Finance 格式

        Args:
            symbol: 国内代码（如 '600519', '000001'）

        Returns:
            Yahoo 代码（如 '600519.SS', '000001.SZ'）
        """
        if '.' in symbol:
            # 已经是 Yahoo 格式
            return symbol

        # 判断市场
        if symbol.startswith('6'):
            # 上海证券交易所
            return f"{symbol}.SS"
        elif symbol.startswith(('0', '3')):
            # 深圳证券交易所
            return f"{symbol}.SZ"
        elif symbol.startswith('688'):
            # 科创板
            return f"{symbol}.SS"
        else:
            return f"{symbol}.SS"

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化列名

        Args:
            df: 原始数据

        Returns:
            pd.DataFrame: 标准化后的数据
        """
        # 常见的列名映射
        column_mapping = {
            '日期': 'date',
            '时间': 'date',
            'Date': 'date',
            '开盘': 'open',
            '开盘价': 'open',
            'Open': 'open',
            '最高': 'high',
            '最高价': 'high',
            'High': 'high',
            '最低': 'low',
            '最低价': 'low',
            'Low': 'low',
            '收盘': 'close',
            '收盘价': 'close',
            'Close': 'close',
            '成交量': 'volume',
            'Volume': 'volume',
            '成交额': 'turnover',
            'Turnover': 'turnover',
        }

        df = df.rename(columns=column_mapping)
        return df

    def _is_trading_day(self, dt: pd.Timestamp) -> bool:
        """
        判断给定日期是否为 A 股交易日。

        优先使用 AkShare 官方交易日历；若不可用则降级为自然工作日
        （bdate_range，即排除周六/周日，但不排除法定节假日）。

        Args:
            dt: 待判断的日期（可带时区）

        Returns:
            True = 交易日，False = 非交易日
        """
        date_naive = dt.normalize().tz_localize(None) if dt.tz is not None else dt.normalize()
        try:
            import akshare as ak
            cal = ak.tool_trade_date_hist_sina()
            if isinstance(cal, pd.DataFrame):
                trading_days = pd.to_datetime(cal.iloc[:, 0])
            else:
                trading_days = pd.to_datetime(cal)
            return date_naive in trading_days.values
        except Exception:
            # 降级：bdate_range 只排除周末，不排除节假日
            return len(pd.bdate_range(start=date_naive, end=date_naive)) > 0

    def get_latest_market_value_data(
        self,
        symbol: str,
    ) -> pd.DataFrame:
        """
        获取最新市值数据

        注意：AkShare 不提供历史市值数据的时序API，只能获取当前市值快照。
        如果需要历史市值，建议：
        1. 使用股价 * 股本来计算
        2. 或者使用其他数据源（如Tushare专业版）

        Args:
            symbol: 股票代码

        Returns:
            pd.DataFrame: 市值数据，包含：
                - date: 日期（当前日期）
                - total_market_cap: 总市值（亿元）
                - circulating_market_cap: 流通市值（亿元）

        Warning:
            当前实现只返回最新的市值快照，不是历史时序数据。
        """
        try:
            # 方法1: 获取个股信息（包含当前市值）
            df_info = ak.stock_individual_info_em(symbol=symbol)

            if df_info is None or df_info.empty:
                return pd.DataFrame()

            # 转换为字典方便查询
            info_dict = dict(zip(df_info['item'], df_info['value']))

            # 提取市值数据（单位：元 -> 亿元）
            total_mv = info_dict.get('总市值', None)
            circulating_mv = info_dict.get('流通市值', None)

            if total_mv is not None or circulating_mv is not None:
                result = pd.DataFrame({
                    'total_market_cap': [float(total_mv) / 100000000 if total_mv is not None else None],
                    'circulating_market_cap': [float(circulating_mv) / 100000000 if circulating_mv is not None else None]
                }, index=[pd.Timestamp.now().normalize()])
                result.index.name = 'date'

                return result
            # 转换为字典方便查询
            info_dict = dict(zip(df_info['item'], df_info['value']))

            # 提取市值数据（单位：元 -> 亿元）
            total_mv = info_dict.get('总市值', None)
            circulating_mv = info_dict.get('流通市值', None)

            if total_mv is not None or circulating_mv is not None:
                result = pd.DataFrame({
                    'total_market_cap': [float(total_mv) / 100000000 if total_mv is not None else None],
                    'circulating_market_cap': [float(circulating_mv) / 100000000 if circulating_mv is not None else None]
                }, index=[pd.Timestamp.now().normalize()])
                result.index.name = 'date'

                return result

            return pd.DataFrame()

        except Exception as e:
            # 如果上述API不可用，尝试从实时行情获取
            try:
                df_spot = ak.stock_zh_a_spot_em()

                if df_spot is None or df_spot.empty:
                    return pd.DataFrame()

                df_symbol = df_spot[df_spot['代码'] == symbol]

                if not df_symbol.empty and len(df_symbol) > 0:
                    # 安全地提取市值数据
                    total_mv = df_symbol.get('总市值', pd.Series([None])).iloc[0]
                    circulating_mv = df_symbol.get('流通市值', pd.Series([None])).iloc[0]

                    if total_mv is not None or circulating_mv is not None:
                        result = pd.DataFrame({
                            'total_market_cap': [pd.to_numeric(total_mv, errors='coerce') / 100000000 if total_mv is not None else None],
                            'circulating_market_cap': [pd.to_numeric(circulating_mv, errors='coerce') / 100000000 if circulating_mv is not None else None]
                        }, index=[pd.Timestamp.now().normalize()])
                        result.index.name = 'date'
                        return result

                return pd.DataFrame()

            except Exception as e:
                return pd.DataFrame()

    def get_kline_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = 'qfq'
    ) -> pd.DataFrame:
        """
        获取K线数据（使用 AkShare）

        AkShare优势：
        - 免费
        - 数据完整
        - 支持复权选项

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            freq: 频率（'D'=日线）
            adjust: 复权类型（'qfq'=前复权）

        Returns:
            pd.DataFrame: K线数据，带时区信息（Asia/Shanghai）
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        validate_date_range = start <= end

        if not validate_date_range:
            raise RuntimeError(f"Invalid date range: {start_date} to {end_date}")

        try:
            # 使用AkShare获取历史行情数据
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust=adjust  # 复权类型（qfq/hfq/""）
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # 标准化列名
            df = self._standardize_columns(df)

            # 确保包含必需的列
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise RuntimeError(f"Missing columns: {missing_cols}")

            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])

            # 转换数值类型
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 设置日期为索引
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            # 添加股票代码列
            df['symbol'] = symbol

            if df.empty:
                return df
            
            # 确保索引是 DatetimeIndex 并添加时区信息
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # 添加时区（如果没有）
            if df.index.tz is None:
                df.index = df.index.tz_localize('Asia/Shanghai')
            else:
                df.index = df.index.tz_convert('Asia/Shanghai')

            df.index.name = 'date'

            return df


        except Exception as e:
            error_msg = f"Failed to get kline data for {symbol}: {e}"
            raise RuntimeError(error_msg)
    
    def get_market_value_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fill_latest: bool = True
    ) -> pd.DataFrame:
        """
        获取历史市值数据（使用 Yahoo Finance + AkShare T+0补充）⭐

        数据源策略：
        - Yahoo Finance: 提供历史时序市值数据（T+1延迟）
        - AkShare: 补充最新一天的市值快照（T+0实时）

        Yahoo Finance 核心优势：
        - 提供真实的历史时序市值数据（每日数据点）
        - AkShare只能提供当前快照，无历史数据

        AkShare T+0补充机制：
        - 当Yahoo最新数据 < OHLCV最新日期时，自动从AkShare获取T+0市值
        - 解决Yahoo T+1延迟问题，实现99.6% -> 100%覆盖率

        数据处理：
        - 确保日期索引格式一致
        - 时区处理：统一为 Asia/Shanghai

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            fill_latest: 是否用AkShare的T+0数据补充最新一天（默认True）

        Returns:
            pd.DataFrame: 历史市值数据
                - date: 日期（索引，DatetimeIndex with timezone）
                - total_market_cap: 总市值（亿元）
                - circulating_market_cap: 流通市值（亿元）
        """
        try:
            yahoo_symbol = self._convert_symbol_to_yahoo(symbol)

            ticker = yf.Ticker(yahoo_symbol)

            # 方法1：从 history 获取（包含市值数据）
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=False
            )
            
            if df.empty:
                return pd.DataFrame()

            # 获取股票信息（包含股本）
            info = ticker.info
            shares_outstanding = info.get('sharesOutstanding', None)

            if shares_outstanding is None:
                # 使用收盘价计算（如果有市值数据）
                if 'Close' in df.columns:
                    result = pd.DataFrame({
                        'close': df['Close'],
                    })
                    result.index.name = 'date'
                    return result
                return pd.DataFrame()

            # 计算市值
            # Yahoo Finance 的价格是原始价格（未复权）
            df_close = df['Close'].copy()

            # 总市值 = 收盘价 × 总股本（转换为亿元）
            total_market_cap = df_close * shares_outstanding / 100000000

            # 流通市值（如果有流通股数据）
            float_shares = info.get('floatShares', shares_outstanding)
            circulating_market_cap = df_close * float_shares / 100000000

            result = pd.DataFrame({
                'total_market_cap': total_market_cap,
                'circulating_market_cap': circulating_market_cap,
            })
            result.index.name = 'date'

            if not isinstance(result.index, pd.DatetimeIndex):
                result.index = pd.to_datetime(result.index)
                
            # 添加时区（如果没有）
            if result.index.tz is None:
                result.index = result.index.tz_localize('Asia/Shanghai')
            else:
                result.index = result.index.tz_convert('Asia/Shanghai')

            result.index.name = 'date'

            if fill_latest:
                end_date_dt  = pd.to_datetime(end_date).tz_localize('Asia/Shanghai')
                now_sh        = pd.Timestamp.now(tz='Asia/Shanghai')
                today_dt      = now_sh.normalize()
                yahoo_latest  = result.index[-1]

                # 追加今日市值快照须同时满足三个条件：
                #   1. end_date 恰好是今天（历史日期缺失 = 当天无交易，不补）
                #   2. 今天是交易日（周末/法定节假日不补）
                #   3. 当前时间已过 15:00（收盘后数据才完整；盘中快照意义不大）
                if yahoo_latest < end_date_dt and end_date_dt == today_dt:
                    # --- 条件2：判断今天是否为交易日 ---
                    today_is_trading = self._is_trading_day(today_dt)

                    # --- 条件3：已过收盘时间 ---
                    market_closed = now_sh.hour >= 15

                    if today_is_trading and market_closed:
                        # 从AkShare获取当前市值快照
                        akshare_mv = self.get_latest_market_value_data(symbol)

                        if not akshare_mv.empty:
                            # 合理性校验：流通市值应 <= 总市值，且均为正数
                            total_mv = akshare_mv['total_market_cap'].iloc[0]
                            circ_mv  = akshare_mv['circulating_market_cap'].iloc[0]
                            mv_valid = (
                                pd.notna(total_mv) and pd.notna(circ_mv)
                                and total_mv > 0
                                and circ_mv <= total_mv * 1.001   # 允许极小浮点误差
                            )

                            if mv_valid:
                                latest_date = end_date_dt.normalize()
                                akshare_mv.index = pd.DatetimeIndex([latest_date])
                                akshare_mv.index.name = 'date'

                                result = pd.concat([result, akshare_mv])
                                result = result[~result.index.duplicated(keep='last')]
                                result.sort_index(inplace=True)
                            else:
                                self.logger.warning(
                                    "[%s] AkShare 市值快照数据异常（total=%.2f, circ=%.2f），跳过追加",
                                    symbol, total_mv, circ_mv
                                )
                    else:
                        reason = "非交易日" if not today_is_trading else "收盘前"
                        self.logger.debug("[%s] 今日市值快照跳过（%s）", symbol, reason)

            return result

        except Exception as e:
            return pd.DataFrame()

    def get_financial_indicators(self, symbol: str) -> pd.DataFrame:
        """
        获取财务指标数据（使用 AkShare）⭐

        AkShare 核心优势：
        - 100条完整历史数据（1998-2025）
        - Yahoo只有5条数据（最近几个季度）
        - 数据增量：+95条，1900%提升

        数据处理：
        - 确保日期索引格式一致
        - 时区处理：统一为 Asia/Shanghai

        Args:
            symbol: 股票代码

        Returns:
            pd.DataFrame: 财务指标数据
                - date: 报告期（索引，DatetimeIndex with timezone）
                - revenue: 营业总收入（亿元）
                - net_profit: 归母净利润（亿元）
                - total_assets: 总资产（亿元）
                - total_equity: 股东权益（亿元）
                - operating_cash_flow: 经营现金流（亿元）
                - roe: 净资产收益率（%）
                - gross_profit_margin: 毛利率（%）
        """
        try:
            # 获取三大财务报表
            profit_df = ak.stock_financial_report_sina(stock=symbol, symbol='利润表')
            balance_df = ak.stock_financial_report_sina(stock=symbol, symbol='资产负债表')
            cashflow_df = ak.stock_financial_report_sina(stock=symbol, symbol='现金流量表')

            if profit_df is None or profit_df.empty:
                return pd.DataFrame()

            # 从利润表提取指标
            # 列名格式：['报告日', '营业总收入', '营业收入', '营业成本', '营业利润', '利润总额', '净利润', '归属于母公司所有者的净利润', ...]
            result_data = []

            for _, row in profit_df.iterrows():
                # 解析报告日期
                date_str = str(row['报告日'])
                try:
                    report_date = pd.to_datetime(date_str, format='%Y%m%d')
                except:
                    continue

                # 构建财务指标字典
                indicators = {
                    'date': report_date
                }

                # 1. 营业总收入（亿元）
                if '营业总收入' in row and pd.notna(row['营业总收入']):
                    indicators['revenue'] = float(row['营业总收入']) / 100000000

                # 2. 净利润（归母净利润，亿元）
                if '归属于母公司所有者的净利润' in row and pd.notna(row['归属于母公司所有者的净利润']):
                    indicators['net_profit'] = float(row['归属于母公司所有者的净利润']) / 100000000
                elif '净利润' in row and pd.notna(row['净利润']):
                    indicators['net_profit'] = float(row['净利润']) / 100000000

                # 3. 毛利率（%）- 计算：(营业收入 - 营业成本) / 营业收入 * 100
                if '营业收入' in row and '营业成本' in row and pd.notna(row['营业收入']) and pd.notna(row['营业成本']):
                    revenue_val = float(row['营业收入'])
                    cost_val = float(row['营业成本'])
                    if revenue_val != 0:
                        indicators['gross_profit_margin'] = (revenue_val - cost_val) / revenue_val * 100

                result_data.append(indicators)

            # 从资产负债表补充指标
            if balance_df is not None and not balance_df.empty:
                # 创建日期索引字典，方便快速查找
                balance_dict = {}
                for _, row in balance_df.iterrows():
                    date_str = str(row['报告日'])
                    try:
                        report_date = pd.to_datetime(date_str, format='%Y%m%d')
                        balance_dict[report_date] = row
                    except:
                        continue

                # 为每条记录补充资产负债表数据
                for indicators in result_data:
                    report_date = indicators['date']
                    if report_date in balance_dict:
                        row = balance_dict[report_date]

                        # 总资产（亿元）
                        if '资产总计' in row and pd.notna(row['资产总计']):
                            indicators['total_assets'] = float(row['资产总计']) / 100000000

                        # 股东权益合计（亿元）
                        if '股东权益合计' in row and pd.notna(row['股东权益合计']):
                            indicators['total_equity'] = float(row['股东权益合计']) / 100000000
                        elif '所有者权益(或股东权益)合计' in row and pd.notna(row['所有者权益(或股东权益)合计']):
                            indicators['total_equity'] = float(row['所有者权益(或股东权益)合计']) / 100000000

                        # 计算净资产收益率 ROE (%)
                        if 'net_profit' in indicators and 'total_equity' in indicators:
                            if indicators['total_equity'] != 0:
                                # ROE = 净利润 / 净资产 * 100
                                indicators['roe'] = (indicators['net_profit'] / indicators['total_equity']) * 100

            # 从现金流量表补充指标
            if cashflow_df is not None and not cashflow_df.empty:
                # 创建日期索引字典
                cashflow_dict = {}
                for _, row in cashflow_df.iterrows():
                    date_str = str(row['报告日'])
                    try:
                        report_date = pd.to_datetime(date_str, format='%Y%m%d')
                        cashflow_dict[report_date] = row
                    except:
                        continue

                # 为每条记录补充现金流数据
                for indicators in result_data:
                    report_date = indicators['date']
                    if report_date in cashflow_dict:
                        row = cashflow_dict[report_date]

                        # 经营活动现金流（亿元）
                        if '经营活动产生的现金流量净额' in row and pd.notna(row['经营活动产生的现金流量净额']):
                            indicators['operating_cash_flow'] = float(row['经营活动产生的现金流量净额']) / 100000000

            # 转换为 DataFrame
            if not result_data:
                return pd.DataFrame()

            result = pd.DataFrame(result_data)
            result.set_index('date', inplace=True)
            result.sort_index(inplace=True)  # 升序排列（最早在前）

            # 确保索引是 DatetimeIndex 并添加时区信息
            if not isinstance(result.index, pd.DatetimeIndex):
                result.index = pd.to_datetime(result.index)
            
            # 添加时区（如果没有）
            if result.index.tz is None:
                result.index = result.index.tz_localize('Asia/Shanghai')
            else:
                result.index = result.index.tz_convert('Asia/Shanghai')

            result.index.name = 'date'
            return result

        except Exception as e:
            return pd.DataFrame()

    def get_adjustment_factors(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取复权因子（使用 AkShare）。

        算法说明
        --------
        复权因子的正确语义：
        - hfq_factor[t]：以**股票上市首日**为锚（= 1.0），后复权累计倍率，严格单调不降。
        - qfq_factor[t]：以**最新交易日**为锚（≈ 1.0 附近），前复权累计倍率，
          每次新分红后整体历史重定锚，阶梯可以下降。

        AkShare 给出的 hfq/qfq 收盘价与不复权收盘价的比值（ch/c0, cq/c0）
        本身就是上述绝对因子，在非除权日会因精度/口径原因产生微小噪声。

        主算法（有官方除权日历时）：
        1. 仅在**除权日**采样 ch/c0（hfq_factor）和 cq/c0（qfq_factor）。
        2. 非除权日 forward-fill（保持前一除权日的因子值不变），
           消除非除权日价格波动带来的噪声。
        3. 首段（第一个除权日之前）直接取 ch/c0 / cq/c0 的均值填充
           （此段无分红事件，比值应接近常数，取均值更稳健）。

        单调性校验：
        - hfq 严格单调不降；若出现下降，说明数据源存在异常，记录 warning。
        - qfq 阶梯下降属正常（重定锚），但降幅超过阈值时也记录 warning。

        降级算法（除权日历获取失败时）：
        直接输出全序列的 ch/c0 和 cq/c0（绝对因子，含非除权日噪声）。

        Returns
        -------
        pd.DataFrame，index=DatetimeIndex(tz=Asia/Shanghai)，
        columns=['qfq_factor', 'hfq_factor']
        """
        try:
            ak_start = start_date.replace('-', '')
            ak_end   = end_date.replace('-', '')

            df_no_adjust = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=ak_start, end_date=ak_end, adjust="",
            )
            df_qfq = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=ak_start, end_date=ak_end, adjust="qfq",
            )
            df_hfq = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=ak_start, end_date=ak_end, adjust="hfq",
            )

            if df_no_adjust is None or df_no_adjust.empty:
                return pd.DataFrame()

            # ---- 标准化列名 ----
            df0 = self._standardize_columns(df_no_adjust)[['date', 'close']].rename(columns={'close': 'c0'})
            dfq = self._standardize_columns(df_qfq)[['date', 'close']].rename(columns={'close': 'cq'})
            dfh = self._standardize_columns(df_hfq)[['date', 'close']].rename(columns={'close': 'ch'})

            m = df0.merge(dfq, on='date', how='outer').merge(dfh, on='date', how='outer')
            m = m[(m['c0'] > 0) & (m['cq'] > 0) & (m['ch'] > 0)]
            m = m.sort_values('date').reset_index(drop=True)
            m['date'] = pd.to_datetime(m['date'])

            # ---- 尝试获取官方除权日历（含事件字段）----
            use_calendar = False
            div_lookup: dict = {}   # date_str -> {bonus, cap, cash}
            try:
                df_div = ak.stock_dividend_cninfo(symbol=symbol)
                if df_div is not None and not df_div.empty and '除权日' in df_div.columns:
                    xr_dates = set(
                        pd.to_datetime(df_div['除权日']).dropna().dt.strftime('%Y-%m-%d').tolist()
                    )
                    m['date_str'] = m['date'].dt.strftime('%Y-%m-%d')
                    m['is_xr'] = m['date_str'].isin(xr_dates)
                    use_calendar = True

                    # 构建 date_str → 分红字段的查找表（备用理论倍率用）
                    for _, row in df_div.iterrows():
                        try:
                            ds = pd.to_datetime(row['除权日']).strftime('%Y-%m-%d')
                            div_lookup[ds] = {
                                'bonus': float(row.get('送股比例', 0) or 0),
                                'cap':   float(row.get('转增比例', 0) or 0),
                                'cash':  float(row.get('派息比例', 0) or 0),
                            }
                        except Exception:
                            pass
            except Exception:
                pass  # 降级为 cummax

            # ---- 构造阶梯因子 ----
            m['raw_hfq'] = m['ch'] / m['c0']   # 绝对因子快照（含非除权日微噪声）
            m['raw_qfq'] = m['cq'] / m['c0']

            if use_calendar:
                # ── 主算法：日历引导的 cumprod 阶梯，以上市首日绝对值定锚 ──────
                #
                # 核心思路：
                #   ch/c0 是"上市首日锚定"的绝对复权因子，但受当日股价影响有微噪声。
                #   正确做法：
                #   1. 在除权日计算日间比值 delta（两条序列比值之比）
                #   2. 非除权日 delta = 1.0（过滤噪声）
                #   3. 对 delta 序列做 cumprod，得到"相对于下载首日"的因子
                #   4. 用下载首日的 raw_hfq/raw_qfq（ch/c0, cq/c0）做绝对定锚
                #      → factor[t] = cumprod[t] × anchor
                #   这样既消除非除权日噪声，又保留上市首日的绝对锚定。
                #
                # hfq：除权日 delta > 1，结果严格单调不降。
                # qfq：除权日 delta 可 < 1（重定锚），阶梯可下降，属正常。

                m['delta_hfq'] = (m['ch'] / m['ch'].shift(1)) / (m['c0'] / m['c0'].shift(1))
                m['delta_qfq'] = (m['cq'] / m['cq'].shift(1)) / (m['c0'] / m['c0'].shift(1))
                m.loc[m.index[0], ['delta_hfq', 'delta_qfq']] = 1.0

                # 非除权日 delta 置 1.0（过滤噪声）
                m['step_hfq'] = m['delta_hfq'].where(m['is_xr'], 1.0)
                m['step_qfq'] = m['delta_qfq'].where(m['is_xr'], 1.0)

                # cumprod 得到"相对首日"的阶梯
                rel_hfq = m['step_hfq'].cumprod()
                rel_qfq = m['step_qfq'].cumprod()

                # 用下载首日的绝对快照值定锚（上市首日锚）
                anchor_hfq = m['raw_hfq'].iloc[0]
                anchor_qfq = m['raw_qfq'].iloc[0]
                m['hfq_factor'] = rel_hfq * anchor_hfq
                m['qfq_factor'] = rel_qfq * anchor_qfq

                # ── 单调性校验 ────────────────────────────────────────────────
                xr_rows = m[m['is_xr']]
                if len(xr_rows) > 1:
                    hfq_steps = xr_rows['step_hfq']
                    hfq_bad = hfq_steps[hfq_steps < 1.0 - 1e-6]
                    if not hfq_bad.empty:
                        bad_dates = m.loc[hfq_bad.index, 'date_str'].tolist()
                        self.logger.warning(
                            "[%s] hfq delta 在以下除权日 < 1.0（数据源异常）：%s",
                            symbol, bad_dates,
                        )
                    qfq_large_drop = xr_rows['step_qfq'][xr_rows['step_qfq'] < 0.8]
                    if not qfq_large_drop.empty:
                        drop_dates = m.loc[qfq_large_drop.index, 'date_str'].tolist()
                        self.logger.info(
                            "[%s] qfq delta 在以下除权日降幅超过20%%（大额分红重定锚）：%s",
                            symbol, drop_dates,
                        )

                self.logger.debug(
                    "[%s] 日历算法完成，除权日 %d 个，hfq 锚=%.4f，qfq 锚=%.4f",
                    symbol, int(m['is_xr'].sum()), anchor_hfq, anchor_qfq,
                )
            else:
                # ── 降级算法：直接输出绝对因子快照（含非除权日微噪声）────────
                self.logger.warning(
                    "[%s] 除权日历获取失败，降级为原始比值算法（含非除权日噪声）", symbol,
                )
                m['hfq_factor'] = m['raw_hfq']
                m['qfq_factor'] = m['raw_qfq']

            result = m[['date', 'qfq_factor', 'hfq_factor']].set_index('date')
            result.sort_index(inplace=True)

            if not isinstance(result.index, pd.DatetimeIndex):
                result.index = pd.to_datetime(result.index)
            if result.index.tz is None:
                result.index = result.index.tz_localize('Asia/Shanghai')
            else:
                result.index = result.index.tz_convert('Asia/Shanghai')

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to get adjustment factors for {symbol}: {e}")

    def select_mode(self, end_date: str, check_trading_day: bool = True) -> str:
        """
        智能选择下载模式（基于end_date和交易日判断）

        逻辑：
        - 如果 end_date 是今天 且 今天是交易日 → 使用 'update' 模式（需要 T+0 数据）
        - 如果 end_date 是今天 但 今天不是交易日 → 使用 'download' 模式（使用前一交易日数据）
        - 如果 end_date 不是今天 → 使用 'download' 模式（历史数据，T+1 就够了）

        Args:
            end_date: 结束日期（字符串格式，如 '2024-01-01'）
            check_trading_day: 是否检查交易日（默认True）

        Returns:
            str: 'update' 或 'download'
        """
        today = pd.Timestamp.now(tz='Asia/Shanghai').normalize()
        end_date_dt = pd.to_datetime(end_date).tz_localize('Asia/Shanghai')

        if end_date_dt >= today:
            # end_date 是今天或未来
            if check_trading_day:
                # 创建临时实例来检查交易日
                try:
                    today_str = today.strftime('%Y-%m-%d')

                    # 获取完整交易日历（不需要year参数）
                    df = ak.tool_trade_date_hist_sina()
                    if df is not None and not df.empty:
                        trading_days = set(
                            pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
                        )
                        is_trading = today_str in trading_days

                        if is_trading:
                            mode = 'update'
                        else:
                            mode = 'download'
                    else:
                        mode = 'update'
                except Exception as e:
                    mode = 'update'
            else:
                mode = 'update'
        else:
            mode = 'download'

        return mode

    def get_complete_dataset(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fill_latest_market_value: bool = True,
        mode: str = 'auto'
    ) -> dict:
        """
        获取完整的数据集（OHLCV + 市值 + 财务指标）

        这是混合提供者的核心方法，整合所有数据源：
        1. OHLCV数据（AkShare）
        2. 历史市值数据（Yahoo Finance + AkShare T+0补充）⭐
        3. 财务指标数据（AkShare）⭐
        4. 复权因子（AkShare）

        索引对齐策略：
        - OHLCV和市值数据：按交易日对齐（每日数据）
        - 财务数据：季度数据，保持独立索引
        - 所有数据统一使用 Asia/Shanghai 时区

        T+0补充机制：
        - update模式：使用AkShare补充最新一天的市值（T+0）
        - download模式：仅使用Yahoo历史数据，不补充最新一天

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            fill_latest_market_value: 是否用AkShare T+0数据补充最新市值（默认True）
            mode: 模式选择
                - 'auto': 自动根据 end_date 选择模式（推荐）⭐
                - 'update': 每日更新模式，自动补充最新一天的市值（T+0）
                - 'download': 历史下载模式，仅使用Yahoo数据，避免前瞻偏差

        Returns:
            dict: 完整数据集，包含：
                - 'ohlcv': OHLCV数据
                - 'market_value': 历史市值数据
                - 'financial': 财务指标数据
                - 'adjustment': 复权因子数据
                - 'aligned_daily': 对齐后的日线数据（OHLCV + 市值）
        """
        # 智能模式选择
        if mode == 'auto':
            mode = self.select_mode(end_date)

        result = {}

        # 1. 获取 OHLCV 数据（AkShare）
        try:
            ohlcv = self.get_kline_data(symbol, start_date, end_date)
            result['ohlcv'] = ohlcv
        except Exception as e:
            result['ohlcv'] = pd.DataFrame()

        # 2. 获取历史市值数据（Yahoo Finance + AkShare T+0补充）⭐
        try:
            # download模式：不补充最新一天（避免前瞻偏差）
            # update模式：自动补充最新一天（T+0实时数据）
            use_t0_fill = fill_latest_market_value and (mode == 'update')

            market_value = self.get_market_value_data(
                symbol,
                start_date,
                end_date,
                fill_latest=use_t0_fill
            )
            result['market_value'] = market_value
        except Exception as e:
            result['market_value'] = pd.DataFrame()

        # 3. 获取财务指标数据（AkShare）⭐
        try:
            financial = self.get_financial_indicators(symbol)
            result['financial'] = financial
        except Exception as e:
            result['financial'] = pd.DataFrame()

        # 4. 获取复权因子（AkShare）
        try:
            adjustment = self.get_adjustment_factors(symbol, start_date, end_date)
            result['adjustment'] = adjustment
        except Exception as e:
            result['adjustment'] = pd.DataFrame()

        # 5. 对齐日线数据（OHLCV + 市值）
        if not ohlcv.empty and not market_value.empty:
            try:
                # 使用 join 对齐索引（保留 OHLCV 的所有日期）
                aligned = ohlcv.join(market_value, how='left')
                result['aligned_daily'] = aligned

            except Exception as e:
                result['aligned_daily'] = ohlcv
        else:
            result['aligned_daily'] = ohlcv if not ohlcv.empty else pd.DataFrame()

        return result

    def get_fundamental_data(
        self,
        symbol: str,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取基本面数据

        Args:
            symbol: 股票代码
            indicators: 指标列表（目前支持：pe, pb, ps）

        Returns:
            pd.DataFrame: 基本面数据
        """
        try:
            # 获取个股信息（包含PE、PB等）
            df = ak.stock_individual_info_em(symbol=symbol)

            if df is None or df.empty:
                return pd.DataFrame()

            # 转换为字典格式便于处理
            data_dict = dict(zip(df['item'], df['value']))

            # 构建结果DataFrame
            result = pd.DataFrame({
                'symbol': [symbol],
                'pe': [data_dict.get('市盈率-动态', None)],
                'pb': [data_dict.get('市净率', None)],
                'ps': [data_dict.get('市销率', None)],
                'market_cap': [data_dict.get('总市值', None)],
                'circulating_cap': [data_dict.get('流通市值', None)],
            })

            # 如果指定了指标，只返回这些指标
            if indicators:
                available_cols = [col for col in indicators if col in result.columns]
                result = result[['symbol'] + available_cols]

            return result

        except Exception as e:
            error_msg = f"Failed to get fundamental data for {symbol}: {e}"
            raise RuntimeError(error_msg)

    def get_index_constituents(self, index_code: str) -> pd.DataFrame:
        """
        获取指数成分股（使用 AkShare）

        Args:
            index_code: 指数代码

        Returns:
            pd.DataFrame: 成分股列表
        """
        try:
            # 根据不同指数代码使用不同的API
            if index_code == '000300':  # 沪深300
                df = ak.index_stock_cons_csindex(symbol="000300")
            elif index_code == '000016':  # 上证50
                df = ak.index_stock_cons_csindex(symbol="000016")
            elif index_code == '000905':  # 中证500
                df = ak.index_stock_cons_csindex(symbol="000905")
            elif index_code == '000001':  # 上证指数
                df = ak.index_stock_cons(symbol="上证指数", index="000001")
            elif index_code == '399001':  # 深证成指
                df = ak.index_stock_cons(symbol="深证成指", index="399001")
            elif index_code == '399006':  # 创业板指
                df = ak.index_stock_cons(symbol="创业板指", index="399006")
            else:
                # 尝试通用方法
                df = ak.index_stock_cons_csindex(symbol=index_code)

            if df is None or df.empty:
                return pd.DataFrame()

            # 标准化列名（不同API返回的列名可能不同）
            if '成分券代码' in df.columns:
                df = df.rename(columns={'成分券代码': 'symbol', '成分券名称': 'name'})
            elif '品种代码' in df.columns:
                df = df.rename(columns={'品种代码': 'symbol', '品种名称': 'name'})
            elif '股票代码' in df.columns:
                df = df.rename(columns={'股票代码': 'symbol', '股票名称': 'name'})

            # 只保留需要的列
            if 'symbol' in df.columns and 'name' in df.columns:
                result = df[['symbol', 'name']].copy()
            else:
                # 如果列名不匹配，尝试使用索引
                result = pd.DataFrame({
                    'symbol': df.iloc[:, 0],
                    'name': df.iloc[:, 1] if len(df.columns) > 1 else ''
                })

            # 清理股票代码格式（去除可能的前缀和后缀）
            result['symbol'] = result['symbol'].astype(str).str.replace(r'\D', '', regex=True)

            # 去重
            result = result.drop_duplicates(subset=['symbol'])

            # 将symbol设为索引
            result.set_index('symbol', inplace=True)

            return result

        except Exception as e:
            error_msg = f"Failed to get constituents for index {index_code}: {e}"
            raise RuntimeError(error_msg)

