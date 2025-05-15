import pandas as pd

df_ratio = pd.read_csv('~/Downloads/sku_ratio.csv', skip_blank_lines=True)
df = pd.read_csv('~/Downloads/周年庆 - 需下单.csv', skip_blank_lines=True)

df_merge = df.merge(df_ratio, how='left', left_on='goods_id', right_on='goods_id')
print(df_merge.columns)
df_merge['sku_num'] = df_merge['第一批预估下单总量'] * df_merge['sku_in_goods_ratio']
print(df_merge['sku_num'].sum())
print(df['第一批预估下单总量'].sum())
# df_merge[['goods_id', '是否海运', '当前折扣', '秋冬', '会员', '最新闪购', '降价是否', '图片',
#        'Unnamed: 8', '商品链接', '开发类型', '站点首次上架日期', '站点当前上架天数', '当前是否卖库存', '上期综合',
#        '上次折扣', '提报折扣', '商家提报', '采购价CNY', '选择', '类型', '改价时间', '吊牌价(Local)',
#        '销量', '日均销量', '闪购天数6天', '营销预计(5.22-6.4日)14天', '第一批预估下单总量',
#        'sale_sku_id', 'skc_id',
#        'sku_in_goods_ratio','sku_num']].to_csv('~/Downloads/周年庆 - 需下单_sku2.csv')
