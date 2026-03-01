import time
import akshare as ak
from tqdm import tqdm
import pandas as pd

# ---------------------------------------------------------------
# 第一步：构建 三级行业名 → 一/二级行业名 的映射表
# sw3.上级行业 = 二级行业名，sw2.上级行业 = 一级行业名
# ---------------------------------------------------------------
print('正在获取申万行业层级信息...')
sw1_info = ak.sw_index_first_info()
sw2_info = ak.sw_index_second_info()
sw3_info = ak.sw_index_third_info()

# sw2名称 → sw1名称
sw2_to_sw1 = dict(zip(sw2_info['行业名称'], sw2_info['上级行业']))
# sw3名称 → sw2名称（即 sw3.上级行业）
sw3_to_sw2 = dict(zip(sw3_info['行业名称'], sw3_info['上级行业']))

print(f'一级行业: {len(sw1_info)} 个，二级行业: {len(sw2_info)} 个，三级行业: {len(sw3_info)} 个')

# ---------------------------------------------------------------
# 第二步：遍历全部三级行业，拉取成分股
# ---------------------------------------------------------------
print('\n正在拉取各行业成分股（约258个三级行业）...')
rows = []
failed_industries = []

for _, ind_row in tqdm(sw3_info.iterrows(), total=len(sw3_info), desc='拉取行业成分股'):
    sw3_code = ind_row['行业代码']
    sw3_name = ind_row['行业名称']
    sw2_name = sw3_to_sw2.get(sw3_name, '')
    sw1_name = sw2_to_sw1.get(sw2_name, '')

    try:
        time.sleep(0.3)
        cons = ak.sw_index_third_cons(symbol=sw3_code)
        if cons is None or cons.empty:
            continue
        for _, stock_row in cons.iterrows():
            raw_code = str(stock_row['股票代码'])
            symbol = raw_code.split('.')[0].zfill(6)   # 600519.SH → 600519
            rows.append({
                'symbol'   : symbol,
                'name'     : str(stock_row.get('股票简称', '')),
                'sw1_name' : sw1_name,
                'sw2_name' : sw2_name,
                'sw3_name' : sw3_name,
                'sw3_code' : sw3_code,
            })
    except Exception as e:
        failed_industries.append((sw3_code, sw3_name, str(e)))

industry_df = (
    pd.DataFrame(rows)
    .drop_duplicates(subset='symbol', keep='first')
    .reset_index(drop=True)
)

print(f'\n✅ 获取完成：{len(industry_df)} 只股票有行业信息')
if failed_industries:
    print(f'⚠️  {len(failed_industries)} 个行业获取失败：')
    for code, name, err in failed_industries[:5]:
        print(f'   {code} {name}: {err}')

industry_df.head(5)
industry_df.to_csv('data/industry_mapping.csv', index=False)