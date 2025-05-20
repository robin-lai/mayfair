import pandas as pd

df_ratio = pd.read_csv('~/Downloads/sku_ratio.csv', skip_blank_lines=True)
df = pd.read_csv('~/Downloads/5.20日周年庆下单2.csv', skip_blank_lines=True)

df_merge = df.merge(df_ratio, how='left', left_on='goods_id', right_on='goods_id')
print(df_merge.columns)
df_merge['sku_num'] = df_merge['num'] * df_merge['sku_in_goods_ratio']
print(df_merge['sku_num'].sum())
print(df['num'].sum())
df_merge[['goods_id', 'num',
       'sale_sku_id', 'skc_id',
       'sku_in_goods_ratio','sku_num']].to_csv('~/Downloads/5.20日周年庆下单2_sku.csv')
