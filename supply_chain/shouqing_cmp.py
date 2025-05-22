import pandas as pd
df_raw = pd.read_csv('~/Downloads/shouqing_skc_raw.csv', skip_blank_lines=True)
df_beihuo = pd.read_csv('~/Downloads/shouqing_raw_skc_v2.csv', skip_blank_lines=True)

df_merge = df_beihuo.merge(df_raw, how='left', left_on='skc_id', right_on='SKC ID')
print(df_merge.columns)
df_merge[['ds', 'goods_id', 'skc_id', 'skc_gap', 'lt', 'buffer_day',
       'deliver_limit_day', 'skc_sale3_per_day',
       'skc_sale7_per_day', 'skc_orders_per_day', 'skc_orders',
       'goods_predict_diff_ratio', 'skc_lable', 'total_stock',
       '入库来源单号', 'CN入库数量', '海运在途件数', '海运在途占比', '全球售罄率' ]].to_csv('~/Downloads/shouqing_case_ana_skc.csv', index=False)
