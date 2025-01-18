# encoding:utf-8
import multiprocessing
import os
from random import shuffle
import pprint
import random
import argparse
import time

import pandas as pd
from pyarrow import parquet
import pickle
import datetime
import numpy as np
import sys
from pathlib import Path
# print(sys.path)
# sys.path.append(str(Path(__file__).absolute().parent.parent.parent.parent))
# sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
print(sys.path)
import gc

import math

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    num = math.ceil(len(lst) / n)
    for i in range(0, len(lst), num):
        yield lst[i:i + num]

item_bhv_user_list_file = './%s_item_bhv_user_list.pkl'
user_bhv_item_list_file = './%s_user_bhv_item_list.pkl'
user_bhv_num_file = './%s_user_bhv_num.pkl'
item_bhv_num_file = './%s_item_bhv_num.pkl'
trig_item_list_file = './%s_trig_item_list_part_%s.pkl'
item_info_file = './%s_item_info.pkl'
pklfile = './swing_rec_%s_part_%s.pkl'
round_num = 5


def process(lines, c, part, sample_num=None):
    user_bhv_item_list = {}
    user_bhv_num = {}
    item_bhv_user_list = {}
    item_bhv_num = {}
    item_info = {}

    for line in lines:
        u, itm, clk, cat2, cat3, leaf = line[0], line[1], line[2], line[3], line[4], line[5],
        if itm not in item_info.keys():
            item_info[itm] = (cat2, cat3,leaf)
        # u, itm, clk = line[0], line[1], line[2]
        if u in user_bhv_item_list.keys():
            user_bhv_item_list[u].add(itm)
        else:
            user_bhv_item_list[u] = set([itm])
        if itm in item_bhv_user_list.keys():
            item_bhv_user_list[itm].add(u)
        else:
            item_bhv_user_list[itm] = set([u])
        # count
        if u in user_bhv_num.keys():
            user_bhv_num[u] += 1
        else:
            user_bhv_num[u] = 1
        if itm in item_bhv_num.keys():
            item_bhv_num[itm] += 1
        else:
            item_bhv_num[itm] = 1
    print('data desc: user num %s, item num:%s'%(len(user_bhv_num.keys()), len(item_bhv_user_list.keys())))
    print('dump data into file')
    # sample
    user_bhv_item_list_new = {}
    for u, v in user_bhv_item_list.items():
        if len(v) >= 600:
            v_s = random.sample(list(v), 600)
            user_bhv_item_list_new[u] = set(v_s)
        else:
            user_bhv_item_list_new[u] = v
    item_bhv_user_list_new = {}
    for itm, u in item_bhv_user_list.items():
        if len(u) >= 700:
            u_s = random.sample(list(u), 700)
            item_bhv_user_list_new[itm] = set(u_s)
        else:
            item_bhv_user_list_new[itm] = u

    # for u, n in user_bhv_num.items():
    #     if n >= 600:
    #         user_bhv_num[u] = 600
    #
    # for itm, n in item_bhv_num.items():
    #     if n >= 700:
    #         item_bhv_num[itm] = 700

    with open(item_bhv_user_list_file%(c), 'wb') as fout:
        pickle.dump(item_bhv_user_list_new, fout)
    with open(user_bhv_item_list_file%(c), 'wb') as fout:
        pickle.dump(user_bhv_item_list_new, fout)
    with open(user_bhv_num_file%(c), 'wb') as fout:
        pickle.dump(user_bhv_num, fout)
    with open(item_bhv_num_file%(c), 'wb') as fout:
        pickle.dump(item_bhv_num, fout)
    with open(item_info_file % (c), 'wb') as fout:
        pickle.dump(item_info, fout)

    item_list = []
    hot_item_num = 0
    for k, v in item_bhv_num.items():
        item_list.append(k)
        if v > 1000:
            hot_item_num += 1
    batch = part
    print('item_list_raw:%s  item_list_filter:%s  hot_item_num:%s'%(len(item_bhv_num.keys()), len(item_list), hot_item_num))
    shuffle(item_list)
    item_batch = list(chunks(item_list, batch))
    for i, ele in enumerate(item_batch):
        with open(trig_item_list_file % (c, i), 'wb') as fout:
            if sample_num is None:
                pickle.dump(ele, fout)
            else:
                pickle.dump(ele[0:sample_num], fout)
        print('batch size:', len(ele))
    print('%s : %s process deal data len:%s'%(str(batch), str(len(item_batch)), str(len(item_list))))


user_debias = True
item_debias = True
out_file = ''
def swing(*args):
    with open(args[0], 'rb') as fin:
        trig_itm_list = pickle.load(fin)
    print('O(n):', len(trig_itm_list))
    # trig_itm_list = args[0]
    out_file = args[1]
    c = args[2]
    s3_file = args[3]
    pkl_file = args[4]
    beta = args[5]
    alph = args[6]
    ubeta = args[7]
    with open(item_bhv_user_list_file%(c), 'rb') as fin:
        item_bhv_user_list = pickle.load(fin)
    with open(user_bhv_item_list_file%(c), 'rb') as fin:
        user_bhv_item_list = pickle.load(fin)
    with open(user_bhv_num_file%(c), 'rb') as fin:
        user_bhv_num = pickle.load(fin)
    with open(item_bhv_num_file%(c), 'rb') as fin:
        item_bhv_num = pickle.load(fin)
    ret = {}
    n = 0
    N = len(trig_itm_list)
    st0 = time.time()
    st = time.time()
    for trig_itm in trig_itm_list:
        swing = {}
        user_sample = list(item_bhv_user_list[trig_itm])
        u_num = len(user_sample)
        if u_num < 2:
            continue
        # if u_num >= 700:
        #     user_sample = random.sample(user, 700)
        # else:
        #     user_sample = user
        # u_num = len(user_sample)
        # print('common user num:', u_num)
        n += 1
        for i in range(0, u_num-1):
            for j in range(i + 1, u_num):
                # print('user a', user[i], 'user b', user[j])
                common_items = user_bhv_item_list[user_sample[i]] & user_bhv_item_list[user_sample[j]]
                for tgt_item in common_items:
                    if trig_itm == tgt_item:
                        continue
                    if user_debias and item_debias:
                        score = round((1 / math.pow(user_bhv_num[user_sample[i]], ubeta))
                                      * (1 / math.pow(user_bhv_num[user_sample[j]], ubeta))
                                      * (1 / math.pow(item_bhv_num[tgt_item], beta))
                                      * (1 / (alph + (len(common_items)))), round_num)
                    elif user_debias:
                        score = round((1 / user_bhv_num[user_sample[i]]) * (1 / user_bhv_num[user_sample[j]]) * (
                                    1 / (alph + (len(common_items)))), round_num)
                    elif item_debias:
                        score = round((1 / math.pow(item_bhv_num[tgt_item], beta)) * (1 / (alph + (len(common_items)))), round_num)
                    else:
                        score = round((1 / (alph + (len(common_items)))), round_num)
                    if tgt_item in swing:
                        swing[tgt_item] = round(swing[tgt_item] +  score, round_num)
                    else:
                        swing[tgt_item] = score
        if len(swing.keys()) < 1:
            continue
        ret[trig_itm] = [(k, v) for k, v in swing.items()]
        if n % 50 == 0:
            ed = time.time()
            print('process 50 / %s item cost:%s' % (str(N), str(ed - st)))
            st = time.time()
    print('swing process done, cost:', time.time() - st0)

    print('write swing result to file:', out_file)
    with open(pkl_file, 'wb') as fout:
        pickle.dump(ret, fout)
    with open(out_file, 'w') as fout:
        lines = []
        for trig, tgt in ret.items():
            tgt = [e for e in tgt if float(e[1]) > 0]
            tgt.sort(key=lambda x: x[1], reverse=True)
            vs = []
            row_n = 100
            for ele in tgt:
                row_n -= 1
                if row_n == 0:
                    break
                vs.append(str(ele[0]) + chr(4) + str(ele[1]))
            line = (c + chr(4) + str(trig) + chr(1) + chr(2).join(vs) + '\n')
            lines.append(line)
        fout.writelines(lines)
    os.system('aws s3 cp %s %s' % (out_file, s3_file))



def get_data_from_s3(raw_file,item_feature):
    st = time.time()
    print('begin read parquet data from file:', raw_file)
    pt = parquet.read_table(raw_file)
    m = {}
    for uuid, goods_id, clk_num, country_code in zip(pt['uuid'], pt['goods_id'], pt['clk_num'], pt['country_code']):
        country_code = country_code.as_py()
        # t = (uuid.as_py(), goods_id.as_py(), clk_num.as_py(),cat2,cat3,leaf)
        if goods_id.as_py() in item_feature:
            goods_info = item_feature[goods_id.as_py()]
            t = (uuid.as_py(), goods_id.as_py(), clk_num.as_py(),goods_info["cate_level2_name"],goods_info["cate_level3_name"],goods_info["cate_name"])
        else:
            t = (uuid.as_py(), goods_id.as_py(), clk_num.as_py(),"", "", "")
        if country_code in m:
            m[country_code].append(t)
        else:
            m[country_code] = [t]
    ed = time.time()
    print('end read file,cost:%s'%(str(ed - st)))
    print('data describe:', '*' * 50)
    for k, v in m.items():
        print('country:%s lines:%s' % (k, len(v)))
    return m

def get_mock_data():
    m = {'cn': [
        ("A", 'z', 1, 'cn'),
        ("A", 'p', 1, 'cn'),
        ("A", 't', 1, 'cn'),
        ("A", 'r', 1, 'cn'),
        ("A", 'h', 1, 'cn'),
        ("B", 'h', 1, 'cn'),
        ("B", 't', 1, 'cn'),
        ("B", 'r', 1, 'cn'),
        ("B", 'p', 1, 'cn'),
        ("C", 'h', 1, 'cn'),
        ("C", 'p', 1, 'cn'),
        ("C", 'y', 1, 'cn'),
        ("C", 'q', 1, 'cn'),
        ("D", 'h', 1, 'cn'),
        ("D", 'q', 1, 'cn'),
        ("E", 'h', 1, 'cn'),
        ("E", 'q', 1, 'cn'),
        ("E", 'o', 1, 'cn'),
        ("E", 'x', 1, 'cn'),
    ]}
    return m

def get_test_data():
    st = time.time()
    file = './cn_rec_detail_recall_ui_relation.txt'
    print('begin read parquet data from file:', file)
    m = {}
    ll = []
    with open(file, 'r') as fout:
        lines = fout.readlines()
        for line in lines:
            ll.append([e.strip('\n') for e in line.split(' ')])
    m['cn'] = ll
    ed = time.time()
    print('end read file,cost:%s'%(str(ed - st)))
    print('data describe:', '*' * 50)
    for k, v in m.items():
        print('country:%s lines:%s' % (k, len(v)))
    return m

def get_data(args, country, item_feature):
    st = time.time()
    in_file = args.in_file
    if args.flag == 's3':
        m = get_data_from_s3(in_file, item_feature)
    elif args.flag == 'mock':
        m = get_mock_data()
    elif args.flag == 'sample':
        m = get_test_data()
    else:
        print('unknown flag:', args.flag)
    ed = time.time()
    print('step 1 get_date done cost:', str(ed - st))
    # preprocess
    v = m[country]
    process(v, country, args.p, args.sample_num)

def swing_result_ana(args, country, p):
    d = {}
    for i in range(p):
        with open(pklfile % (country, i), 'rb') as fin:
            d.update(pickle.load(fin))
    print('swing result merge item num:', len(d.keys()))
    with open(item_info_file%country, 'rb') as fin:
        item_info = pickle.load(fin)
    with open(item_bhv_num_file%country, 'rb') as fin:
        item_num = pickle.load(fin)

    dd = {}
    ll = []
    stat = {'is_cat2_rel_ratio': 0, 'is_cat3_rel_ratio': 0, 'is_leaf_rel_ratio': 0}
    topn = 100
    n = len(d.keys())
    for itm, v in d.items():
        trig_t = item_info[itm]
        v = [e for e in v if float(e[1]) > 0]
        v.sort(key=lambda x: x[1], reverse=True)
        cat2_c, cat3_c, leaf_c = 0,0,0
        vs = v if len(v) <= 100 else v[0:topn]
        tgt_list_tmp = []
        for ele in vs:
            tgt_t = item_info[ele[0]]
            tmp_l = [str(ele[0]), str(item_num[ele[0]]), str(ele[1]) , str(tgt_t[0]), str(tgt_t[1]) ,str(tgt_t[2])]
            tgt_list_tmp.append(','.join(tmp_l))
            if trig_t[0] == tgt_t[0]:
                cat2_c += 1
            if trig_t[1] == tgt_t[1]:
                cat3_c += 1
            if trig_t[2] == tgt_t[2]:
                leaf_c += 1
        ll.append([itm,item_num[itm],trig_t[0], trig_t[1], trig_t[2], '|'.join(tgt_list_tmp)])
        c = len(vs)
        if c > 0:
            dd[itm] = [c, item_num[itm],  cat2_c / c, cat3_c / c, leaf_c /c ]
            stat['is_cat2_rel_ratio'] += cat2_c / c
            stat['is_cat3_rel_ratio'] += cat3_c / c
            stat['is_leaf_rel_ratio'] +=  leaf_c /c

    df = pd.DataFrame(ll, columns=['trig', 'num', 'cat2', 'cat3', 'leaf', 'tgt-itm-num-score-cat2-cat3-leaf'])
    local_file = './' + args.swing_ana_file
    swing_s3_file = args.s3_dir + args.swing_ana_file
    df.to_csv(local_file)
    os.system('aws s3 cp %s %s' % (local_file, swing_s3_file))
    print('swing_s3_file:', swing_s3_file)
    for k, v in stat.items():
        stat[k] = v / n
    print('swing result relate stat:')
    print(stat)

def get_item_feature(file):
    ret = {}
    pt = parquet.read_table(file).to_pylist()
    for e in pt:
        ret[int(e['goods_id'])] = e
    return ret

def main(args, item_feature):
    # get data
    country = 'Savana_IN'
    if args.pipeline != 'ana':
        st = time.time()
        get_data(args, country, item_feature)
        gc.collect()
        # swing
        st = time.time()
        outfile = './swing_rec_%s_part_%s'
        s3_file = args.s3_dir + 'swing_rec_%s_part_%s'
        proc_list = [multiprocessing.Process(target=swing, args=[trig_item_list_file%(country, i), outfile%(country,i), country, s3_file%(country, i), pklfile%(country, i), args.beta, args.alph, args.ubeta]) for i in range(args.p)]
        [p.start() for p in proc_list]
        [p.join() for p in proc_list]
        fail_cnt = sum([p.exitcode for p in proc_list])
        if fail_cnt:
            raise ValueError('Failed in %d process.' % fail_cnt)
        print('step swing done cost:', str(time.time() - st))
        swing_result_ana(args, country, args.p)
    else:
        swing_result_ana(args, country, args.p)

if __name__ == '__main__':
    # 'https://help.aliyun.com/zh/pai/use-cases/improved-swing-similarity-calculation-algorithm' 700, 500截断逻辑
    # 每个用户的最长序列长度，如果超过该长度会对最近进行截断保留, 600
    # 每个物品使用多少个用户的点击序列来计算k近邻
    parser = argparse.ArgumentParser(
        prog='swing',
        description='swing-args',
        epilog='swing-help')
    parser.add_argument('--flag',default='s3')
    parser.add_argument('--v',default='')
    parser.add_argument('--pipeline',default='swing')
    parser.add_argument('--beta', type=float, default=0.7)
    parser.add_argument('--ubeta', type=float, default=0.5)
    parser.add_argument('--alph', type=float, default=1.0)
    parser.add_argument('--p',type=int, default=7)
    parser.add_argument('--sample_num',type=int, default=None)
    parser.add_argument('--pre_ds', type=str, default=(datetime.date.today() - datetime.timedelta(days=2)).strftime('%Y%m%d'))
    parser.add_argument('--in_file', type=str, default='s3://warehouse-algo/rec/cn_rec_detail_recall_ui_relation%s/ds=%s')
    parser.add_argument('--s3_dir', type=str, default='s3://warehouse-algo/rec/recall/cn_rec_detail_recall_i2i_for_redis%s/item_user_debias_%s_%s_%s_%s/')
    parser.add_argument('--swing_ana_file', type=str, default='swing_result%s_%s.csv')
    parser.add_argument('--item', default='s3://warehouse-algo/rec/cn_rec_detail_feature_item_base/ds=%s/')
    args = parser.parse_args()
    if ',' in args.pre_ds:
        ds_range = args.pre_ds
        print('process range:', ds_range)
        for pre_ds in ds_range.split(','):
            args = parser.parse_args()
            args.pre_ds = pre_ds
            args.in_file = args.in_file % (args.v, args.pre_ds)
            args.s3_dir = args.s3_dir % (args.v, args.pre_ds, str(args.alph), str(args.beta), str(args.ubeta))
            args.swing_ana_file = args.swing_ana_file % (args.v, args.pre_ds)
            print('s3_dir', args.s3_dir)
            print('in_file', args.in_file)
            print('swing_ana_file', args.swing_ana_file)
            st = time.time()
            item_feature = get_item_feature(args.item % args.pre_ds)
            main(args, item_feature)
            ed = time.time()
            # job_d = {"start_time": str(st), "end_time": str(ed), "cost":str(ed-st)}
            # add_job_monitor('tfr', job_d)
            print('final cost:', ed - st)
    else:
        args.in_file = args.in_file % (args.v, args.pre_ds)
        args.s3_dir = args.s3_dir % (args.v, args.pre_ds, str(args.alph), str(args.beta), str(args.ubeta))
        args.swing_ana_file = args.swing_ana_file % (args.v, args.pre_ds)
        print('s3_dir', args.s3_dir)
        print('in_file', args.in_file)
        print('swing_ana_file', args.swing_ana_file)
        st = time.time()
        item_feature = get_item_feature(args.item % args.pre_ds)
        main(args, item_feature)
        ed = time.time()
        # job_d = {"start_time": str(st), "end_time": str(ed), "cost":str(ed-st)}
        # add_job_monitor('tfr', job_d)
        print('final cost:', ed-st)

# nohup python -u swing_multi_process.py --pre_ds=20250106 --beta=0.6 done
# nohup python -u swing_multi_process.py --pre_ds=20250106 --beta=0.7 done
# nohup python -u swing_multi_process.py --pre_ds=20250107 --beta=0.7 --ubeta=0.5 --alph=1.0 --p=10  cpu vs mem: 6:8 r5.4x  6440s
# nohup python -u swing_multi_process.py --pre_ds=20250108,20250109,20250110,20250111,20250112,20250113,20250114,20250115,20250116 --beta=0.7 --ubeta=0.5 --alph=1.0 --p=10 done
# nohup python -u swing_multi_process.py --pre_ds=20241229,20241230,20241231,20250101,20250102,20250103,20250104,20250105 --beta=0.7 --ubeta=0.5 --alph=1.0 --p=10 run
# nohup python -u swing_multi_process.py --pre_ds=20241217,20241218,20241219,20241220,20241221,20241222,20241223 --beta=0.7 --ubeta=0.5 --alph=1.0 --p=10 run
# nohup python -u swing_multi_process.py --pre_ds=20241224,20241225,20241226,20241227 --beta=0.7 --ubeta=0.5 --alph=1.0 --p=10 wait
# error ds 20241228
# 10内存会满
