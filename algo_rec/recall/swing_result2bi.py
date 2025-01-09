import pandas as pd
from pyarrow import parquet
import pyarrow as pa
import argparse
import time
import os

import gc

def main(args):

    pt = parquet.read_table(args.item_file).to_pylist()
    d = {}
    for e in pt:
        d[e['goods_id']] = {'goods_pic_url': e['goods_pic_url'], 'goods_name': e['goods_name']}
    gc.collect()
    dd = {'trig_goods_id': [], 'trig_goods_name': [], 'num': [], 'cat2': [], 'cat3': [], 'leaf': [], 'leaf_cn':[], 'trig_pic_url': [],
          'tgt_goods_id': [], 'tgt_goods_name': [], 'tgt_score': [], 'tgt_pic_url': [], 'tgt_num': [], 'tgt_cat2': [],
          'tgt_cat3': [], 'tgt_leaf': [], 'tgt_leaf_cn':[]}
    local_file = args.swing_result.split('/')[-1]
    os.system("aws s3 cp %s %s" % (args.swing_result, local_file))
    df = pd.read_csv(local_file)
    swing_ll = df.to_dict(orient='records')

    leaf_local = args.leaf_info.split('/')[-1]
    os.system("aws s3 cp %s %s" % (args.leaf_info, leaf_local))
    leaf_df = pd.read_csv(leaf_local)
    leaf_ll = leaf_df.to_dict(orient='records')
    map_d = {}
    for e in leaf_ll:
        map_d[e['cate_id']] = e

    st = time.time()
    for idx, e in enumerate(swing_ll):
        tgt = e['tgt-itm-num-score-cat2-cat3-leaf']
        if ',' not in str(tgt):
            continue
        # print(tgt)
        if idx % 1000 == 0:
            ed = time.time()
            print('process %s of %s cost %s' % ((idx / 1000) * 1000, len(swing_ll), ed-st))
            st = time.time()
        if e['trig'] in d:
            trig_n = d[e['trig']]['goods_name']
            trig_url = d[e['trig']]['goods_pic_url']
        else:
            trig_n = ''
            trig_url = ''

        for tt_str in tgt.split('|'):
            tt = tt_str.split(',')
            if int(tt[0]) not in d:
                tgt_pic = ''
                tgt_good_name = ''
            else:
                tgt_pic = d[int(tt[0])]['goods_pic_url']
                tgt_good_name = d[int(tt[0])]['goods_name']

            dd['trig_goods_id'].append(int(e['trig']))
            dd['trig_goods_name'].append(str(trig_n))
            dd['num'].append(int(e['num']))
            dd['cat2'].append(str(e['cat2']))
            dd['cat3'].append(str(e['cat3']))
            dd['leaf'].append(str(e['leaf']))
            dd['leaf_cn'].append(str(map_d[e['leaf']]['cate_name_cn']))
            dd['trig_pic_url'].append(str(trig_url))
            dd['tgt_goods_id'].append(int(tt[0]))
            dd['tgt_goods_name'].append(str(tgt_good_name))
            dd['tgt_score'].append(float(tt[2]))
            dd['tgt_pic_url'].append(str(tgt_pic))
            dd['tgt_num'].append(int(tt[1]))
            dd['tgt_cat2'].append(str(tt[3]))
            dd['tgt_cat3'].append(str(tt[4]))
            dd['tgt_leaf'].append(str(tt[5]))
            dd['tgt_leaf_cn'].append(str(map_d[str(tt[5])]['cate_name_cn']))

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
    parser.add_argument('--leaf_info', default='s3://warehouse-algo/rec/leafname_map_cn.csv')
    args = parser.parse_args()
    st = time.time()
    main(args)
    ed = time.time()
    print('cost ', ed - st)