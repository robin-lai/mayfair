# encoding:utf-8
import multiprocessing
import os
from random import shuffle
import pprint
import argparse
import time
from pyarrow import parquet
import pickle
import numpy as np
import sys


import math

from tensorflow.python.distribute.multi_process_runner import manager


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    c = math.ceil(len(lst) / n)
    for i in range(0, len(lst), c):
        yield lst[i:i + c]

# item_bhv_user_list_file = './%s_item_bhv_user_list.pkl'
# user_bhv_item_list_file = './%s_user_bhv_item_list.pkl'
# user_bhv_num_file = './%s_user_bhv_num.pkl'
# item_bhv_num_file = './%s_item_bhv_num.pkl'

manager = multiprocessing.Manager()
user_bhv_item_list_m = manager.dict()
user_bhv_num_m = manager.dict()
item_bhv_user_list_m = manager.dict()
item_bhv_num_m = manager.dict()

def process(lines, c):
    user_bhv_item_list = {}
    user_bhv_num = {}
    item_bhv_user_list = {}
    item_bhv_num = {}
    for line in lines:
        u, itm, clk = line[0], line[1], line[2]
        if u in user_bhv_item_list.keys():
            user_bhv_item_list[u].append(itm)
        else:
            user_bhv_item_list[u] = [itm]
        if itm in item_bhv_user_list.keys():
            item_bhv_user_list[itm].append(u)
        else:
            item_bhv_user_list[itm] = [u]
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

    # for k, v in user_bhv_item_list.items():
    st = time.time()
    user_bhv_item_list_m.update(user_bhv_item_list)
    # for k, v in item_bhv_user_list.items():
    item_bhv_user_list_m.update(item_bhv_user_list)
    # for k, v in user_bhv_num.items():
    user_bhv_num_m.update(user_bhv_num)
    # for k, v in item_bhv_num.items():
    item_bhv_num_m.update(item_bhv_num)
    ed = time.time()
    print('parse manager cost:', str(ed-st))

    print('user_bhv_item_list', user_bhv_item_list)
    print('user_bhv_item_list', user_bhv_item_list)
    print('user_bhv_num', user_bhv_num)
    print('item_bhv_num', item_bhv_num)

    # with open(item_bhv_user_list_file%(c), 'wb') as fout:
    #     pickle.dump(item_bhv_user_list, fout)
    # with open(user_bhv_item_list_file%(c), 'wb') as fout:
    #     pickle.dump(user_bhv_item_list, fout)
    # with open(user_bhv_num_file%(c), 'wb') as fout:
    #     pickle.dump(user_bhv_num, fout)
    # with open(item_bhv_num_file%(c), 'wb') as fout:
    #     pickle.dump(item_bhv_num, fout)


alph = 1
user_debias = True
out_file = './tmp/swing_in_result_part_%s.txt'
s3_file = 's3://algo-sg/rec/cn_rec_detail_recall_i2i_for_redis/'
def swing(proc, item_batch_dict_m, swing_ret_m):
    trig_itm_list = item_batch_dict_m[proc]
    print('O(n):', len(trig_itm_list))
    # out_file = args[1]
    # c = args[2]
    # s3_file = args[3]
    # with open(item_bhv_user_list_file%(c), 'rb') as fin:
    #     item_bhv_user_list = pickle.load(fin)
    # with open(user_bhv_item_list_file%(c), 'rb') as fin:
    #     user_bhv_item_list = pickle.load(fin)
    # with open(user_bhv_num_file%(c), 'rb') as fin:
    #     user_bhv_num = pickle.load(fin)
    n = 0
    N = len(trig_itm_list)
    st0 = time.time()
    st = time.time()
    lines = []
    ret = {}
    for trig_itm in trig_itm_list:
        swing = {}
        user = item_bhv_user_list_m[trig_itm]
        u_num = len(user)
        if u_num < 2:
            continue
        # print('common user num:', u_num)
        n += 1
        for i in range(0, u_num-1):
            for j in range(i + 1, u_num):
                # print('user a', user[i], 'user b', user[j])
                common_items = set(user_bhv_item_list_m[user[i]]) & set(user_bhv_item_list_m[user[j]])
                common_items = common_items - set(trig_itm)
                for tgt_item in common_items:
                    if user_debias:
                        score = round((1 / user_bhv_num_m[user[i]]) * (1 / user_bhv_num_m[user[j]]) * (
                                    1 / (alph + (len(common_items)))), 4)
                    else:
                        score = round((1 / (alph + (len(common_items)))), 4)
                    if tgt_item in swing:
                        swing[tgt_item] = round(swing[tgt_item] +  score, 4)
                    else:
                        swing[tgt_item] = score
        swing_ll = [(k, v) for k, v in swing.items()]
        row_n = 30
        swing_ll.sort(key=lambda x: x[1], reverse=True)
        tmp_ll = []
        for ele in swing_ll:
            row_n -= 1
            if row_n == 0:
                break
            tmp_ll.append(ele[0] + chr(4) + str(ele[1]))
            line = ("in" + chr(4) + trig_itm + chr(1) + chr(2).join(tmp_ll) + '\n')
            lines.append(line)
        ret[trig_itm].extend(tmp_ll)
        if n % 50 == 0:
            ed = time.time()
            print('process 50 / %s item cost:%s' % (str(N), str(ed - st)))
            st = time.time()
    swing_ret_m.update(ret)
    print('swing_ret_m keys num:', len(swing_ret_m.keys()))
    out_file_proc = out_file % proc
    print('write swing result2file:', out_file_proc)
    with open(out_file_proc, 'w') as fout:
        fout.writelines(lines)
    os.system('aws s3 cp %s %s' % (out_file_proc, s3_file + out_file_proc))
    print('swing process done, cost:', time.time() - st0)

    # print('write swing result to file:', out_file)
    # row_n = 30
    # with open(out_file, 'w') as fout:
    #     lines = []
    #     for trig, tgt in ret.items():
    #         tgt.sort(key=lambda x: x[1], reverse=True)
    #         vs = []
    #         for ele in tgt:
    #             row_n -= 1
    #             if row_n == 0:
    #                 break
    #             vs.append(ele[0] + chr(4) + str(ele[1]))
    #         line = (c + chr(4) + trig + chr(1) + chr(2).join(vs) + '\n')
    #         lines.append(line)
    #     fout.writelines(line)
    # os.system('aws s3 cp %s %s' % (out_file, s3_file))



def get_data_from_s3(raw_file):
    st = time.time()
    print('begin read parquet data from file:', raw_file)
    pt = parquet.read_table(raw_file)
    m = {}
    for uuid, goods_id, clk_num, country_code in zip(pt['uuid'], pt['goods_id'], pt['clk_num'], pt['country_code']):
        country_code = country_code.as_py()
        t = (uuid.as_py(), goods_id.as_py(), clk_num.as_py())
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

def main(args):
    # get data
    st = time.time()
    in_file = args.in_file
    if args.data == 's3':
        m = get_data_from_s3(in_file)
    elif args.data == 'mock':
        m = get_mock_data()
    elif args.data == 'sample':
        m = get_test_data()
    else:
        print('unknown data:',args.data)
    ed = time.time()
    print('step 1 get_date done cost:', str(ed-st))
    print("data", m)

    # preprocess
    st = time.time()
    for country, v in m.items():
        print('process country:', country)
        # if country != 'IN':
        #     continue
        process(v, country)
        print('step 2 preprocess done cost:', str(time.time() - st))
        # swing
        st = time.time()
        # with open(item_bhv_num_file%(country), 'rb') as fin:
        #     item_bhv_num = pickle.load(fin)
        item_list = []
        hot_item_num = 0
        for k, v in item_bhv_num_m.items():
            if v > 3000:
                hot_item_num += 1
            else:
                item_list.append(k)
        batch = args.p
        print('item_list_raw:%s  item_list_filter:%s  hot_item_num:%s'%(len(item_bhv_num_m.keys()), len(item_list), hot_item_num))
        shuffle(item_list)
        item_batch = list(chunks(item_list, batch))
        item_batch_dict = {}
        item_batch_dict_m = manager.dict()
        swing_ret_m = manager.dict()
        for proc, ele in enumerate(item_batch):
            item_batch_dict[proc] = ele
        item_batch_dict_m.update(item_batch_dict)

        print('batch size:', len(ele))
        print('%s : %s process deal data len:%s'%(str(batch), str(len(item_batch)), str(len(item_list))))
        proc_list = []
        for k, v in item_batch_dict.items():
            proc_list.append(multiprocessing.Process(target=swing, args=(k,item_batch_dict_m, swing_ret_m)))
        [p.start() for p in proc_list]
        [p.join() for p in proc_list]
        fail_cnt = sum([p.exitcode for p in proc_list])
        if fail_cnt:
            raise ValueError('Failed in %d process.' % fail_cnt)
        print('step swing done cost:', str(time.time() - st))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='swing',
        description='swing-args',
        epilog='swing-help')
    parser.add_argument('--data',default='mock')
    parser.add_argument('--p',type=int, default=1)
    parser.add_argument('--s3_dir', type=str, default='s3://algo-sg/rec/cn_rec_detail_recall_i2i_for_redis/')
    parser.add_argument('--in_file', type=str, default='s3://algo-sg/rec/cn_rec_detail_recall_ui_relation/ds=20241118')
    args = parser.parse_args()
    main(args)
