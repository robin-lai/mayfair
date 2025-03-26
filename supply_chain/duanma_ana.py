import pandas as pd
from datetime import datetime

in_file = '~/Downloads/goods_sellable_v4.csv'
# out_file = './goods_sellable_v4_wid_near_all.csv'
out_file = 'goods_sellable_v4_wid_near.csv'
df = pd.read_csv(in_file)
debug = False
debug = True
if debug:
    goods_ids = [1239312]
else:
    goods_ids = set(df['goods_id'].tolist())
# print(goods_ids)

not_valid_value_n = 0
not_valid_value_type2_n = 0
ll = []
for id in goods_ids:
    id_list = df[df['goods_id'] == id].to_dict(orient='records')
    # print(df_id.head(10))
    # id_list.sort(key=lambda x: x['biz_date'])
    id_list = sorted(id_list, key=lambda x: datetime.strptime(x['biz_date'], '%Y-%m-%d'), reverse=True)
    if len(set([e['is_sellable'] for e in id_list])) == 1:
        not_valid_value_type2_n += 1
        continue
    print([(e['is_sellable'],e['biz_date']) for e in id_list])
    n = len(id_list)
    wid_pre,wid_now, wid_suf = [], [], []
    wid_pre_date,wid_now_date, wid_suf_date = [], [], []
    p_pre, p_now, p_suf = 0, 0, 0
    # 在当前时间点，往后找，找到第一个断码区间
    while p_now < n:
        if id_list[p_now]['is_sellable'] == '断码':
            break
        else:
            p_now += 1
    p_now_r = p_now
    while p_now_r < n:
        if id_list[p_now_r]['is_sellable'] == '非断码':
            break
        else:
            wid_now.append(id_list[p_now_r])
            wid_now_date.append(id_list[p_now_r]['biz_date'])
            p_now_r += 1
    # print(p_now)
    # print(p_now_r)

    # 在第一个断码区间，前后各找一个非断码区间
    while p_pre < p_now and p_pre < n:
        if id_list[p_pre]['is_sellable'] == '非断码':
            wid_pre.append(id_list[p_pre])
            wid_pre_date.append(id_list[p_pre]['biz_date'])
            p_pre += 1
        else:
            break

    p_suf = p_now_r
    while p_suf < n:
        if id_list[p_suf]['is_sellable'] == '断码':
            break
        else:
            wid_suf.append(id_list[p_suf])
            wid_suf_date.append(id_list[p_suf]['biz_date'])
            p_suf += 1

    print(f'wid_pre:{len(wid_pre)}, wid_now:{len(wid_now)}, wid_suf:{len(wid_suf)}')
    # print()
    if len(wid_pre) == 0 or len(wid_now) == 0 or len(wid_suf) == 0:
        not_valid_value_n += 1
        continue
    import numpy as np
    def fun(ll,id, date, suffix, name=['show_pv', 'show_uv', 'click_pv', 'click_uv', 'add_cart_pv', 'add_cart_uv', 'add_wish_pv', 'add_wish_uv','confirmed_ord_uv']):
        d = {}
        d['goods_id'] = id
        d['win'] = len(ll)
        d['date'] = date
        d['status'] = suffix
        if len(ll) == 0:
            return d
        for nm in name:
            d[nm] = round(np.average([e[nm] for e in ll]),1)
        d['uv_ctr'] = round(d['click_uv'] / d['show_uv'],4)
        d['pv_ctr'] = round(d['click_pv'] / d['show_pv'],4)
        d['uv_cvr'] = round(d['confirmed_ord_uv'] / d['click_uv'], 4)
        return d
    def fun2(l):
        if len(l) == 0:
            return ''
        sorted_dates = sorted(l, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
        return sorted_dates[0] + '-' + sorted_dates[-1]
    d_suf = fun(wid_suf,id, fun2(wid_suf_date), suffix='有货')
    d_now = fun(wid_now,id, fun2(wid_now_date), suffix='断码')
    d_pre = fun(wid_pre,id, fun2(wid_pre_date), suffix='补货后')
    # three row
    ll.append(d_suf)
    ll.append(d_now)
    ll.append(d_pre)
    ll.append({})
    # print(ll)
if debug:
    for ele in ll:
        print(ele)
else:
    pd.DataFrame(ll).to_csv(out_file)
    # print(d)
print(f'goods_ids_n:{len(goods_ids)}')
print(f'找不到符合条件的三个区间goods_id数量:{not_valid_value_n}')
print(f'没有断货的goods_id数量:{not_valid_value_type2_n}')
print(f'合法的goods_id数量: {len(ll) / 3}')

