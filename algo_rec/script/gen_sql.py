
sql_line = """concat_ws(',' , percentile_approx(%s, ARRAY(%s))) as pp_%s,"""
cols = 'pv_1d,ipv_1d,cart_1d,wish_1d,pay_1d,pctr_1d,pcart_1d,pwish_1d,pcvr_1d,pv_3d,ipv_3d,cart_3d,wish_3d,pay_3d,pctr_3d,pcart_3d,pwish_3d,pcvr_3d,pv_5d,ipv_5d,cart_5d,wish_5d,pay_5d,pctr_5d,pcart_5d,pwish_5d,pcvr_5d,pv_7d,ipv_7d,cart_7d,wish_7d,pay_7d,pctr_7d,pcart_7d,pwish_7d,pcvr_7d,pv_14d,ipv_14d,cart_14d,wish_14d,pay_14d,pctr_14d,pcart_14d,pwish_14d,pcvr_14d,pv_30d,ipv_30d,cart_30d,wish_30d,pay_30d,pctr_30d,pcart_30d,pwish_30d,pcvr_30d'
sql_lines = []
pp = '0.05,0.1,0.15,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0'
for col in cols.split(','):
    print(sql_line % (col, pp, col))