import argparse
import os
# encoding:utf-8
import os
import json
import tensorflow as tf
import sys
import time
import multiprocessing
from multiprocessing import shared_memory
from pyarrow import parquet
import argparse
import boto3
from random import shuffle
import numpy as np
import traceback
import datetime
import pickle

debug = False

s3_cli = boto3.client('s3')
BUCKET = 'warehouse-algo'
s3_buk = "s3://warehouse-algo/"
s3_obj = "rec/"
local_data = "/home/sagemaker-user/mayfair/algo_rec/data/"
import math


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    c = math.ceil(len(lst) / n)
    for i in range(0, len(lst), c):
        yield lst[i:i + c]


def get_i2i(i2i_part, i2i_s3, i2i_file):
    i2i_d = {}
    i2i_file_ll = []
    for i in range(i2i_part):
        s3_file = i2i_s3 + i2i_file % str(i)
        local_file = './' + i2i_file % str(i)
        os.system('rm %s' % local_file)
        os.system('aws s3 cp %s %s' % (s3_file, local_file))
        i2i_file_ll.append(local_file)
    for file in i2i_file_ll:
        with open(file, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                k, v = line.split(chr(1))
                vs = []
                for tt in v.split(chr(2)):
                    vs.append(tt.split(chr(4)))
                i2i_d[k] = vs
    print('read i2i end, num:', len(i2i_d.keys()))
    return i2i_d


def get_u2cart_wish(txt_pt):
    pt = parquet.read_table(txt_pt).to_pylist()
    u2cart_wish_d = {}
    for e in pt:
        u2cart_wish_d[e['uuid']] = {str(tt):1 for tt in e['goods_list']}
    if debug:
        print('get_u2cart_wish')
        for k in u2cart_wish_d.keys()[0:10]:
            print(k, u2cart_wish_d[k])

    return u2cart_wish_d


def get_hot_i2leaf(txt_dir):
    local_txt_dir = './' + txt_dir
    os.system('aws s3 cp --recursive %s %s' % (txt_dir, local_txt_dir))

    hot_i2leaf_d = {}
    for filename in os.listdir(local_txt_dir):
        if filename.endswith(".txt"):  # Filter .txt files
            file_path = os.path.join(local_txt_dir, filename)
            with open(file_path, "r") as infile:
                lines = infile.readlines()
                for line in lines:
                    k, v = line.split(chr(1))
                    hot_i2leaf_d[str(k.split('|')[1])] = {str(tt): 1 for tt in [e.split(chr(4)[0]) for e in v.split(chr(2))]}
    if debug:
        print('hot_i2leaf_d')
        for k in hot_i2leaf_d.keys()[0:10]:
            print(k, hot_i2leaf_d[k])
    return hot_i2leaf_d


def get_site_hot(txt_dir):
    local_txt_dir = './' + txt_dir
    os.system('aws s3 cp --recursive %s %s' % (txt_dir, local_txt_dir))

    site_hot_d = {}
    for filename in os.listdir(local_txt_dir):
        if filename.endswith(".txt"):  # Filter .txt files
            file_path = os.path.join(local_txt_dir, filename)
            with open(file_path, "r") as infile:
                lines = infile.readlines()
                for line in lines:
                    k, v = line.split(chr(1))
                    site_hot_d[k] = { str(tt):1 for tt in [e.split(chr(4)[0]) for e in v.split(chr(2))][0:100]}
    if debug:
        print('site_hot_d')
        for k in site_hot_d.keys()[0:1]:
            print(k, site_hot_d[k])
    return site_hot_d


def run_multi_process(func, args, i2i_d, u2cart_wish_d, hot_i2leaf_d, site_hot_d):
    path_pt = s3_buk + s3_obj + args.dir_pt + args.ds
    path_pt_suffix = s3_obj + args.dir_pt + args.ds
    path_tfr_local = local_data + args.dir_tfr + args.ds
    path_tfr_local_base = local_data + args.dir_tfr
    path_tfr_s3 = s3_buk + s3_obj + args.dir_tfr + args.ds
    os.system('rm -rf %s' % path_tfr_local_base)
    os.system('mkdir -p %s' % path_tfr_local)
    # get files
    paginator = s3_cli.get_paginator('list_objects_v2')
    print('key:', path_pt_suffix)
    page_iter = paginator.paginate(Bucket=BUCKET, Prefix=path_pt_suffix)
    file_list = [[v['Key'] for v in page.get('Contents', [])] for page in page_iter][0]
    if args.sample_num is not None:
        file_list = file_list[0:args.sample_num]
    file_suffix_list = [v.split('/')[-1] for v in file_list]
    print('file list in dir', file_list)
    print('file_suffix_list:', file_suffix_list)
    # batch
    shuffle(file_suffix_list)
    file_batch = list(chunks(file_suffix_list, args.thread))
    # file_batch = np.array_split(file_suffix_list, args.thread)
    args_list = []
    for ll in file_batch:
        l1, l2, l3 = [], [], []
        for file in ll:
            l1.append(path_pt + '/' + file)
            l2.append(path_tfr_local + '/' + file)
            l3.append(path_tfr_s3 + '/' + file)
        args_list.append([l1, l2, l3])
    print('args_list top ', args_list)
    # multiprocess
    # create sha mem
    # 序列化嵌套字典
    i2i_s = pickle.dumps(i2i_d)
    i2i_shm_size = len(i2i_s)
    i2i_shm = shared_memory.SharedMemory(create=True, size=i2i_shm_size)
    i2i_shm.buf[:i2i_shm_size] = i2i_s  # 写入数据
    print(f"Shared memory created. Name: {i2i_shm.name}, Size: {i2i_shm_size} bytes")

    u2cart_wish_s = pickle.dumps(u2cart_wish_d)
    u2cart_wish_shm_size = len(u2cart_wish_s)
    u2cart_wish_shm = shared_memory.SharedMemory(create=True, size=u2cart_wish_shm_size)
    u2cart_wish_shm.buf[:u2cart_wish_shm_size] = u2cart_wish_s  # 写入数据
    print(f"Shared memory created. Name: {u2cart_wish_shm.name}, Size: {u2cart_wish_shm_size} bytes")

    hot_i2leaf_s = pickle.dumps(hot_i2leaf_d)
    hot_i2leaf_shm_size = len(hot_i2leaf_s)
    hot_i2leaf_shm = shared_memory.SharedMemory(create=True, size=hot_i2leaf_shm_size)
    hot_i2leaf_shm.buf[:hot_i2leaf_shm_size] = hot_i2leaf_s  # 写入数据
    print(f"Shared memory created. Name: {hot_i2leaf_shm.name}, Size: {hot_i2leaf_shm_size} bytes")

    site_hot_s = pickle.dumps(site_hot_d)
    site_hot_shm_size = len(site_hot_s)
    site_hot_shm = shared_memory.SharedMemory(create=True, size=site_hot_shm_size)
    site_hot_shm.buf[:site_hot_shm_size] = site_hot_s  # 写入数据
    print(f"Shared memory created. Name: {site_hot_shm.name}, Size: {site_hot_shm_size} bytes")

    proc_list = [multiprocessing.Process(target=func, args=(
    args[0], args[1], args[2], proc_id, i2i_shm.name, i2i_shm_size, u2cart_wish_shm.name, u2cart_wish_shm_size,
    hot_i2leaf_shm.name, hot_i2leaf_shm_size, site_hot_shm.name, site_hot_shm_size)) for proc_id, args in
                 enumerate(args_list)]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]
    fail_cnt = sum([p.exitcode for p in proc_list])
    if fail_cnt > 0:
        print('process fail cnt:', fail_cnt)
        # raise ValueError('Failed in %d process.' % fail_cnt)


def gen_fts(path_pt_list, path_tfr_local_list, path_tfr_s3_list, proc_id, i2i_shm_n
            , i2i_shm_size, u2cart_wish_shm_n, u2cart_wish_shm_size,
            hot_i2leaf_shm_n, hot_i2leaf_shm_size, site_hot_shm_n, site_hot_shm_size):
    def is_tgt_in_recall(ll, i2i_d, tgt_id):
        tmp_d = {}
        n = 20
        for e in ll:
            tt = e.split(1)
            if len(tt) > 2:
                trig = tt[1]
                if trig not in tmp_d:
                    tmp_d[trig] = 1
                    n -= 1
                    if n == 0:
                        break
                    if trig in i2i_d:
                        if tgt_id in i2i_d[trig]:
                            return (True, i2i_d[trig][tgt_id])
        return (False, 0.0)

    i2i_shm = shared_memory.SharedMemory(name=i2i_shm_n)
    i2i_s = bytes(i2i_shm.buf[:i2i_shm_size])
    i2i_d = pickle.loads(i2i_s)  # 反序列化为嵌套字典

    u2cart_wish_shm = shared_memory.SharedMemory(name=u2cart_wish_shm_n)
    u2cart_wish_s = bytes(u2cart_wish_shm.buf[:u2cart_wish_shm_size])
    u2cart_wish_d = pickle.loads(u2cart_wish_s)  # 反序列化为嵌套字典

    hot_i2leaf_shm = shared_memory.SharedMemory(name=hot_i2leaf_shm_n)
    hot_i2leaf_s = bytes(hot_i2leaf_shm.buf[:hot_i2leaf_shm_size])
    hot_i2leaf_d = pickle.loads(hot_i2leaf_s)  # 反序列化为嵌套字典

    site_hot_shm = shared_memory.SharedMemory(name=site_hot_shm_n)
    site_hot_s = bytes(site_hot_shm.buf[:site_hot_shm_size])
    site_hot_d = pickle.loads(site_hot_s)  # 反序列化为嵌套字典

    try:
        # 从共享内存中读取序列化数据
        for pt_file, tfr_local_file, tfr_s3_file in zip(path_pt_list, path_tfr_local_list, path_tfr_s3_list):
            pt = parquet.read_table(pt_file).to_pylist()
            for tt in pt:
                feature = {}
                mt = []
                mt_w = []
                main_goods = tt['main_goods_id']
                tgt_id = tt['goods_id']
                seq_on = tt['seq_on']
                if main_goods in i2i_d:
                    if tgt_id in i2i_d[main_goods]:
                        feature['mt_i2i_main'] = 1
                        feature['mt_i2i_main_score'] = i2i_d[main_goods][tgt_id]
                        mt.append('i2i_main')
                        mt_w.append(i2i_d[main_goods][tgt_id])
                if seq_on == "":
                    continue
                else:
                    js = dict(json.loads(seq_on))
                if 'highLevelSeqList' in js:
                    ret = is_tgt_in_recall(js['highLevelSeqList'], i2i_d, tgt_id)
                    if ret[0]:
                        feature['mt_i2i_long'] = 1
                        feature['mt_i2i_long_score'] = ret[1]
                        mt.append('i2i_long')
                if 'lowerLevelSeqList' in js:
                    ret = is_tgt_in_recall(js['lowerLevelSeqList'], i2i_d, tgt_id)
                    if ret[0]:
                        feature['mt_i2i_short'] = 1
                        feature['mt_i2i_short_score'] = ret[1]
                        mt.append('i2i_short')
    finally:
        # 关闭共享内存
        i2i_shm.close()


def main(args):
    i2i_d = get_i2i(args.i2i_part, args.i2i_s3, args.i2i_file)
    u2cart_wish_d = get_u2cart_wish(args.u2cart_wish_file)
    hot_i2leaf_d = get_hot_i2leaf(args.hot_i2leaf)
    site_hot_d = get_site_hot(args.site_hot)

    run_multi_process(gen_fts, args, i2i_d, u2cart_wish_d, hot_i2leaf_d, site_hot_d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gentfr',
        description='gentfr',
        epilog='gentfr-help')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--ds', type=str,
                        default='ds=%s' % (datetime.date.today() - datetime.timedelta(days=2)).strftime('%Y%m%d'))
    parser.add_argument('--range', type=str, default='')
    parser.add_argument('--i2i_s3',
                        default='s3://warehouse-algo/rec/recall/cn_rec_detail_recall_i2i_for_redis%s/item_user_debias_%s/')
    parser.add_argument('--i2i_file', default='swing_rec_Savana_IN_part_%s')
    parser.add_argument('--i2i_part', type=int, default=7)
    parser.add_argument('--u2cart_wish_file',
                        default='s3://warehouse-algo/rec/recall/cn_rec_detail_recall_wish_cart2i/')
    parser.add_argument('--hot_i2leaf', default='s3://algo-sg/rec/cn_rec_detail_recall_main_leaf2i_for_redis/')
    parser.add_argument('--site_hot', default='s3://algo-sg/rec/cn_rec_detail_recall_site_hot_for_redis/')
    parser.add_argument('--thread', type=int, default=15)
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--dir_pt', default='cn_rec_detail_sample_v20_savana_in/')
    parser.add_argument('--dir_tfr', default='cn_rec_detail_sample_v20_savana_in_tfr/')
    args = parser.parse_args()
    debug = args.debug
    main(args)
