import pandas as pd
from pyarrow import parquet
import pyarrow as pa
import argparse
import time

import gc

def main(args):
    pt = parquet.read_table(args.item_file).to_pylist()
    d = {}
    for e in pt:
        d[e['goods_id']] = {'goods_pic_url': e['goods_pic_url'], 'goods_name': e['goods_name']}
    gc.collect()
    dd = {'trig_goods_id': [], 'trig_goods_name': [], 'num': [], 'cat2': [], 'cat3': [], 'leaf': [], 'trig_pic_url': [],
          'tgt_goods_id': [], 'tgt_goods_name': [], 'tgt_score': [], 'tgt_pic_url': [], 'tgt_num': [], 'tgt_cat2': [],
          'tgt_cat3': [], 'tgt_leaf': []}
    df = pd.read_csv(args.swing_result)
    swing_ll = df.to_dict(orient='records')

    st = time.time()
    for idx, e in enumerate(swing_ll):
        tgt = e['tgt-itm-num-score-cat2-cat3-leaf']
        if ',' not in str(tgt):
            continue
        # print(tgt)
        if idx % 1000 == 0:
            ed = time.time()
            print('process %s of %s cost %s' % ((idx / 1000) * 1000, len(swing_ll), st-ed))
            st = time.time()

        for tt_str in tgt.split('|'):
            tt = tt_str.split(',')
            if int(tt[0]) not in d:
                tgt_pic = ''
                tgt_good_name = ''
            else:
                tgt_pic = d[int(tt[0])]['goods_pic_url']
                tgt_good_name = d[int(tt[0])]['goods_name']

            dd['trig_goods_id'].append(int(e['trig']))
            dd['trig_goods_name'].append(str(d[e['trig']]['goods_name']))
            dd['num'].append(int(e['num']))
            dd['cat2'].append(str(e['cat2']))
            dd['cat3'].append(str(e['cat3']))
            dd['leaf'].append(str(e['leaf']))
            dd['trig_pic_url'].append(str(d[e['trig']]['goods_pic_url']))
            dd['tgt_goods_id'].append(int(tt[0]))
            dd['tgt_goods_name'].append(str(tgt_good_name))
            dd['tgt_score'].append(float(tt[2]))
            dd['tgt_pic_url'].append(str(tgt_pic))
            dd['tgt_num'].append(int(tt[1]))
            dd['tgt_cat2'].append(str(tt[3]))
            dd['tgt_cat3'].append(str(tt[4]))
            dd['tgt_leaf'].append(str(tt[5]))

    tb = pa.table(dd)
    parquet.write_table(tb, args.save_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='swing',
        description='swing-args',
        epilog='swing-help')
    # file = './swing_result_20250106.csv'
    parser.add_argument('--item_file', default='s3://warehouse-algo/rec/dim_mf_goods_s3/ds=20250107')
    parser.add_argument('--save_file', default='s3://warehouse-algo/rec/recall/rec_detail_recall_swing_result')
    parser.add_argument('--swing_result', default='s3://warehouse-algo/rec/recall/cn_rec_detail_recall_i2i_for_redis/item_user_debias_20250106/swing_result_20250106.csv')
    args = parser.parse_args()
    st = time.time()
    main(args)
    ed = time.time()
    print('cost ', ed - st)