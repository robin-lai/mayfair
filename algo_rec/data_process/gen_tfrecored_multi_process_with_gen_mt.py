# encoding:utf-8
import os
import json
import tensorflow as tf
from multiprocessing import shared_memory
import pickle
import sys
import time
import multiprocessing
from pyarrow import parquet
import argparse
import boto3
from random import shuffle
import numpy as np
import traceback
from datetime import datetime,date, timedelta
from pympler import asizeof

import math
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
print(sys.path)
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
print(sys.path)
from algo_rec.utils.util import check_s3_file_exists, alert_feishu

BUCKET = 'warehouse-algo'
BUCKET_S3_PREFIX = "s3://warehouse-algo/"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    c = math.ceil(len(lst) / n)
    for i in range(0, len(lst), c):
        yield lst[i:i + c]


def bytes_fea(v_list, n=1, encode=False):
    v_list = v_list if isinstance(v_list, list) else [v_list]
    if len(v_list) > n:
        v_list = v_list[:n]
    elif len(v_list) < n:
        v_list.extend([""] * (n - len(v_list)))
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[bytes(v, encoding="utf8") for v in v_list]))


def ints_fea(v_list):
    v_list = v_list if isinstance(v_list, list) else [v_list]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v or 0) for v in v_list]))


def floats_fea(v_list):
    v_list = v_list if isinstance(v_list, list) else [v_list]
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v or 0) for v in v_list]))


def cross_fea(v1_list, v2_list, n=1):
    v1_list = v1_list if isinstance(v1_list, list) else [v1_list]
    v2_list = v2_list if isinstance(v2_list, list) else [v2_list]
    # cross feature usuannly contains query or title.
    v3_list = ['%s,,%s' % (v1, v2) for v1 in v1_list for v2 in v2_list]
    return bytes_fea(v3_list, n, True)


def build_tfrecord(path_pt_list, path_tfr_local_list, path_tfr_s3_list,proc_id,stat_file,stat_flag,
                   itm_shm_n, itm_shm_size, i2i_shm_n, i2i_shm_size, u2cart_wish_shm_n, u2cart_wish_shm_size,
                   hot_i2leaf_shm_n, hot_i2leaf_shm_size, site_hot_shm_n, site_hot_shm_size,
                   itm_stat_shm_n, itm_stat_shm_size
                   ):
    itm_shm = shared_memory.SharedMemory(name=itm_shm_n)
    itm_s = bytes(itm_shm.buf[:itm_shm_size])
    itm_d = pickle.loads(itm_s)  # 反序列化为嵌套字典

    itm_stat_shm = shared_memory.SharedMemory(name=itm_stat_shm_n)
    itm_stat_s = bytes(itm_stat_shm.buf[:itm_stat_shm_size])
    itm_stat_d = pickle.loads(itm_stat_s)  # 反序列化为嵌套字典

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

    def is_tgt_in_recall(ll, i2i_d, tgt_id):
        tmp_d = {}
        n = 20
        for e in ll:
            tt = e.split(chr(1))
            if len(tt) > 2:
                trig = int(tt[1])
                if trig not in tmp_d:
                    tmp_d[trig] = 1
                    n -= 1
                    if n == 0:
                        break
                    if trig in i2i_d:
                        if tgt_id in i2i_d[trig]:
                            return (True, i2i_d[trig][tgt_id])
        return (False, 0.0)

    def build_mt(tt, feature, stat_d, stat_flag):
        mt = []
        mt_w = []
        main_goods = int(tt['main_goods_id'])
        tgt_id = int(tt['goods_id'])
        main_cate_id = str(tt['main_cate_id'])
        if tgt_id in site_hot_d['Savana_IN']:
            mt.append('hot')
        if main_cate_id in hot_i2leaf_d and tgt_id in hot_i2leaf_d[main_cate_id]:
            mt.append('hot_i2leaf')
        if tt['uuid'] in u2cart_wish_d and tgt_id in u2cart_wish_d[tt['uuid']]:
            mt.append('u2i_f')  # u2icart_wish
        if main_goods in i2i_d:
            if tgt_id in i2i_d[main_goods]:
                mt.append('i2i_main')
                mt_w.append(i2i_d[main_goods][tgt_id])
        seq_on = tt['seq_on']
        if seq_on != '':
            js = dict(json.loads(seq_on))
            if 'highLevelSeqList' in js:
                ret = is_tgt_in_recall(js['highLevelSeqList'], i2i_d, tgt_id)
                if ret[0]:
                    mt.append('i2i_long')
                    mt_w.append(ret[1])
            if 'lowerLevelSeqList' in js:
                ret = is_tgt_in_recall(js['lowerLevelSeqList'], i2i_d, tgt_id)
                if ret[0]:
                    mt.append('i2i_short')
                    mt_w.append(ret[1])

        feature['mt_i2i_main'] = ints_fea([0])
        feature['mt_i2i_main_score'] = floats_fea([0.0])
        feature['mt_i2i_long'] = ints_fea([0])
        feature['mt_i2i_long_score'] = floats_fea([0.0])
        feature['mt_i2i_short'] = ints_fea([0])
        feature['mt_i2i_short_score'] = floats_fea([0.0])
        feature['mt_u2i_f'] = ints_fea([0])
        for ele in zip(mt, mt_w):
            if ele[0] == 'i2i_main':
                feature['mt_i2i_main'] = ints_fea([1])
                feature['mt_i2i_main_score'] = floats_fea([ele[1]])
            if ele[0] == 'i2i_long':
                feature['mt_i2i_long'] = ints_fea([1])
                feature['mt_i2i_long_score'] = floats_fea([ele[1]])
            if ele[0] == 'i2i_short':
                feature['mt_i2i_short'] = ints_fea([1])
                feature['mt_i2i_short_score'] = floats_fea([ele[1]])
            if ele[0] == 'hot':
                feature['mt_hot'] = ints_fea([1])
            if ele[0] == 'hot_i2leaf':
                feature['mt_hot_i2leaf'] = ints_fea([1])
            if ele[0] == 'u2i_f':
                feature['mt_u2i_f'] = ints_fea([1])
        if stat_flag:
            stat_d['s'].append(mt)
        feature['mt'] = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[bytes(v, encoding="utf8") for v in mt]))
        feature['mt_w'] = floats_fea(mt_w)

    def build_seq_on(seq_on, feature):
        if debug:
            print('seq_on', seq_on)
        if seq_on == "":
            js = dict()
        else:
            js = dict(json.loads(seq_on))
        for k, v in js.items():
            if k not in ['lowerLevelSeqList', 'highLevelSeqList']:
                continue
            goods = []
            cate_id_list = []
            cate_name_list = []
            cate_level3_id_list = []
            cate_level3_name_list = []
            cate_level4_id_list = []
            cate_level4_name_list = []
            for token in v:
                tt = token.split(chr(1))
                if len(tt) < 2:
                    break
                e = int(tt[1])
                goods.append(str(e))
                cate_id_list.append(str(itm_d[e]["cate_id"]))
                cate_name_list.append(str(itm_d[e]["cate_name"]))
                cate_level3_id_list.append(str(itm_d[e]["cate_level3_id"]))
                cate_level3_name_list.append(str(itm_d[e]["cate_level3_name"]))
                cate_level4_id_list.append(str(itm_d[e]["cate_level4_id"]))
                cate_level4_name_list.append(str(itm_d[e]["cate_level4_name"]))
            n = 20 if len(goods) >= 20 else len(goods)
            feature[k + 'Goods'] = bytes_fea(goods, n=20)
            feature[k + 'CateId'] = bytes_fea(cate_id_list, n=20)
            feature[k + 'CateName'] = bytes_fea(cate_name_list, n=20)
            feature[k + 'CateId3'] = bytes_fea(cate_level3_id_list, n=20)
            feature[k + 'CateName3'] = bytes_fea(cate_level3_name_list, n=20)
            feature[k + 'CateId4'] = bytes_fea(cate_level4_id_list, n=20)
            feature[k + 'CateName4'] = bytes_fea(cate_level4_name_list, n=20)
            feature[k + '_len'] = ints_fea(n)

    def build_feature(t, feature):
        item_features_string = {"goods_id": "", "cate_id": "", "cate_level1_id": ""
            , "cate_level2_id": "", "cate_level2_name": "",
                                "cate_level3_id": "", "cate_level3_name": "",
                                "cate_level4_id": "", "cate_level4_name": "",
                                "country": "",
                                "main_goods_id": "",
                                "main_cate_level2_id": "", "main_cate_level2_name": "",
                                "main_cate_level3_id": "", "main_cate_level3_name": "",
                                "main_cate_level4_id": "", "main_cate_level4_name": "",
                                "main_cate_id": "", "main_cate_name": "", "pic_id": "",
                                "goods_name": "", "main_color": "",
                                "prop_seaon": "", "prop_length": "",
                                "prop_main_material": "", "prop_pattern": "",
                                "prop_style": "", "prop_quantity": "", "prop_fitness": ""
                                }
        item_stat_double =  {'pctr_1d': -1.0, 'pcart_1d': -1.0, 'pwish_1d': -1.0, 'pcvr_1d': -1.0, 'pctr_3d': -1.0,
                                'pcart_3d': -1.0, 'pwish_3d': -1.0, 'pcvr_3d': -1.0, 'pctr_5d': -1.0, 'pcart_5d': -1.0,
                                'pwish_5d': -1.0, 'pcvr_5d': -1.0, 'pctr_7d': -1.0, 'pcart_7d': -1.0, 'pwish_7d': -1.0,
                                'pcvr_7d': -1.0, 'pctr_14d': -1.0, 'pcart_14d': -1.0, 'pwish_14d': -1.0,
                                'pcvr_14d': -1.0, 'pctr_30d': -1.0, 'pcart_30d': -1.0, 'pwish_30d': -1.0,
                                'pcvr_30d': -1.0}
        item_stat_int = {'pv_1d': -1, 'ipv_1d': -1, 'cart_1d': -1, 'wish_1d': -1, 'pay_1d': -1, 'pv_3d': -1,
                             'ipv_3d': -1, 'cart_3d': -1, 'wish_3d': -1, 'pay_3d': -1, 'pv_5d': -1, 'ipv_5d': -1,
                             'cart_5d': -1, 'wish_5d': -1, 'pay_5d': -1, 'pv_7d': -1, 'ipv_7d': -1, 'cart_7d': -1,
                             'wish_7d': -1, 'pay_7d': -1, 'pv_14d': -1, 'ipv_14d': -1, 'cart_14d': -1, 'wish_14d': -1,
                             'pay_14d': -1, 'pv_30d': -1, 'ipv_30d': -1, 'cart_30d': -1, 'wish_30d': -1, 'pay_30d': -1
                             }
        item_features_int ={"is_rel_cate": 0, "is_rel_cate2": 0, "is_rel_cate3": 0, "is_rel_cate4": 0, "sales_price": 0}
        user_int = {"age": -1, "seq_len": 0, "pos_idx": -1}
        user_string = {"last_login_device": "", "last_login_brand": "", "register_brand": "", "client_type": ""}
        user_seq_string = {"seq_goods_id": [""] * 20, "seq_cate_id": [""] * 20}
        # user_seq_on_string = {"highLevelSeqListGoods": [""] * 20, "highLevelSeqListCateId": [""] * 20,
        #                       "lowerLevelSeqListGoods": [""] * 20, "lowerLevelSeqListCateId": [""] * 20}
        other_string = {"sample_id": "", "uuid": ""}
        other_int = {"is_clk": 0, "is_pay": 0, "is_cart": 0, "is_wish": 0}

        for name in item_features_string.keys():
            feature.update({name: bytes_fea(t[name])})
        for name in user_string.keys():
            feature.update({name: bytes_fea(t[name])})

        for name in other_string.keys():
            feature.update({name: bytes_fea(t[name])})
        for name in item_stat_double.keys():
            if int(t['goods_id']) in itm_stat_d:
                raw_v = itm_stat_d[int(t['goods_id'])][name]
                if raw_v is not None:
                    feature.update({name: floats_fea(raw_v)})
                else:
                    feature.update({name: floats_fea(item_stat_double[name])})
            else:
                feature.update({name: floats_fea(item_stat_double[name])})
        for name in item_features_int.keys():
            feature.update({name: ints_fea(t[name])})
        for name in item_stat_int.keys():
            if int(t['goods_id']) in itm_stat_d:
                raw_v = itm_stat_d[int(t['goods_id'])][name]
                if raw_v is not None:
                    feature.update({name: ints_fea(raw_v)})
                else:
                    feature.update({name: ints_fea(item_stat_int[name])})
            else:
                feature.update({name: ints_fea(item_stat_int[name])})
        for name in user_int.keys():
            feature.update({name: ints_fea(t[name])})
        for name in other_int.keys():
            feature.update({name: ints_fea(t[name])})
        for name in user_seq_string.keys():
            feature.update({name: bytes_fea(t[name], n=20)})

    stat_d = {"sample_id":[], "s":[], "is_clk":[], "is_cart":[], "is_wish":[], "is_pay":[]}
    for pt_file, tfr_local_file, tfr_s3_file in zip(path_pt_list, path_tfr_local_list, path_tfr_s3_list):
        st = time.time()
        pt = parquet.read_table(pt_file).to_pylist()
        ed = time.time()
        print('read ptpath:%s data cost:%s' % (pt_file, str(ed - st)))
        st = time.time()
        fout_ctr = tf.io.TFRecordWriter(tfr_local_file)
        for t in pt:
            feature = dict()
            # try:
            if int(t['pos_idx']) >= 200:
                continue
            if stat_flag:
                stat_d['sample_id'].append(t['sample_id'])
                stat_d['is_clk'].append(t['is_clk'])
                stat_d['is_cart'].append(t['is_cart'])
                stat_d['is_wish'].append(t['is_wish'])
                stat_d['is_pay'].append(t['is_pay'])
            build_mt(t, feature, stat_d, stat_flag)
            build_feature(t, feature)
            build_seq_on(t['seq_on'], feature)
            sample = tf.train.Example(features=tf.train.Features(feature=feature))
            record = sample.SerializeToString()
            fout_ctr.write(record)
            # except Exception:
            #     print("-" * 60)
            #     traceback.print_exc(file=sys.stdout)
            #     print("-" * 60)
            #     print('data:',t)
            if debug:
                print('features', feature)
        ed = time.time()
        fout_ctr.close()
        print('gen trf done, cost %s' % str(ed - st))
        # upload
        print('upload from %s to %s' % (tfr_local_file, tfr_s3_file))
        os.system('aws s3 cp %s %s' % (tfr_local_file, tfr_s3_file))
        os.system('rm %s' % tfr_local_file)
    if stat_flag:
        local_stat_file = './tmp/%s.pkl'%str(proc_id)
        with open(local_stat_file, 'wb') as fout:
            pickle.dump(stat_d, fout)
        os.system('aws s3 cp %s %s' % (local_stat_file, stat_file + '%s.pkl'%(str(proc_id))))


def get_file_list(args):
    s3_cli = boto3.client('s3')
    s3_obj = "rec/"
    local_data = "/home/sagemaker-user/mayfair/algo_rec/data/"
    path_pt = BUCKET_S3_PREFIX + s3_obj + args.dir_pt
    path_pt_suffix = s3_obj + args.dir_pt
    path_tfr_local = local_data + args.dir_tfr
    path_tfr_local_base = local_data + args.dir_tfr
    path_tfr_s3 = BUCKET_S3_PREFIX + s3_obj + args.dir_tfr
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
    return args_list


def get_i2i(i2i_part, i2i_s3, i2i_file):
    i2i_d = {}
    i2i_file_ll = []
    for i in range(i2i_part):
        s3_file = BUCKET_S3_PREFIX + i2i_s3 + i2i_file % str(i)
        if check_s3_file_exists(BUCKET, i2i_s3 + i2i_file % str(i)):
            local_file = './' + i2i_file % str(i)
            os.system('rm %s' % local_file)
            os.system('aws s3 cp %s %s' % (s3_file, local_file))
            i2i_file_ll.append(local_file)
        else:
            print(f"{s3_file} does not exist.")

    for file in i2i_file_ll:
        with open(file, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                k, v = line.split(chr(1))
                tmp_d = {}
                for tt in v.split(chr(2)):
                    tokens = tt.split(chr(4))
                    if len(tokens) == 2:
                        tmp_d[int(tokens[0])] = tokens[1]
                    # else:
                    #     print('error data:', tt)
                i2i_d[int(k.split(chr(4))[1])] = tmp_d
    print('read i2i end, num:', len(i2i_d.keys()))
    return i2i_d


def get_u2cart_wish(txt_pt):
    pt = parquet.read_table(txt_pt).to_pylist()
    u2cart_wish_d = {}
    for e in pt:
        u2cart_wish_d[e['uuid']] = {tt: 1 for tt in e['goods_list']}
    return u2cart_wish_d


def get_hot_i2leaf(txt_dir):
    local_txt_dir = './tmp/hot_i2leaf/'
    print(f"hot_i2leaf local_txt_dir:{txt_dir}, {local_txt_dir}")
    os.system('rm -r %s' % local_txt_dir)
    os.system('aws s3 cp --recursive %s %s' % (txt_dir, local_txt_dir))

    hot_i2leaf_d = {}
    for filename in os.listdir(local_txt_dir):
        # if filename.endswith(".txt"):  # Filter .txt files
        file_path = os.path.join(local_txt_dir, filename)
        print(f"hot_i2leaf file: {file_path}")
        with open(file_path, "r") as infile:
            lines = infile.readlines()
            for line in lines:
                k, v = line.split(chr(1))
                hot_i2leaf_d[str(k.split('|')[1])] = {int(tt): 1 for tt in [e.split(chr(4))[0] for e in v.split(chr(2))]}
    return hot_i2leaf_d


def get_site_hot(txt_dir):
    local_txt_dir = './tmp/site_hot/'
    print(f"site_hot local_txt_dir:{txt_dir}, {local_txt_dir}")
    os.system('rm -r %s' % local_txt_dir)
    os.system('aws s3 cp --recursive %s %s' % (txt_dir, local_txt_dir))

    site_hot_d = {}
    for filename in os.listdir(local_txt_dir):
        # if filename.endswith(".txt"):  # Filter .txt files
        file_path = os.path.join(local_txt_dir, filename)
        print(f"site_hot file:{file_path}")
        with open(file_path, "r") as infile:
            lines = infile.readlines()
            for line in lines:
                k, v = line.split(chr(1))
                site_hot_d[k] = {int(tt): 1 for tt in [e.split(chr(4))[0] for e in v.split(chr(2))][0:100]}
    return site_hot_d


def get_item_feature(file):
    ret = {}
    pt = parquet.read_table(file).to_pylist()
    for e in pt:
        ret[int(e['goods_id'])] = e
    return ret


def get_item_stat(file):
    ret = {}
    pt = parquet.read_table(file).to_pylist()
    for e in pt:
        if e['goods_id'] is not None:
            ret[int(e['goods_id'])] = e
    return ret


def main(args):
    itm_d = get_item_feature(args.item_file)
    itm_s = pickle.dumps(itm_d)
    itm_shm_size = len(itm_s)
    itm_shm = shared_memory.SharedMemory(create=True, size=itm_shm_size)
    itm_shm.buf[:itm_shm_size] = itm_s  # 写入数据
    print('item_feature size of mem [M]', asizeof.asizeof(itm_d) / 1048576)  #
    print('get item features:', len(itm_d.keys()))

    itm_stat_d = get_item_stat(args.item_stat)
    itm_stat_s = pickle.dumps(itm_stat_d)
    itm_stat_shm_size = len(itm_stat_s)
    itm_stat_shm = shared_memory.SharedMemory(create=True, size=itm_stat_shm_size)
    itm_stat_shm.buf[:itm_stat_shm_size] = itm_stat_s  # 写入数据
    print('item_stat_feature size of mem [M]', asizeof.asizeof(itm_stat_d) / 1048576)  #
    print('get item features:', len(itm_stat_d.keys()))

    i2i_d = get_i2i(args.i2i_part, args.i2i_s3, args.i2i_file)
    print('i2i_d size of mem [M]', asizeof.asizeof(i2i_d) / 1048576)  #
    i2i_s = pickle.dumps(i2i_d)
    i2i_shm_size = len(i2i_s)
    i2i_shm = shared_memory.SharedMemory(create=True, size=i2i_shm_size)
    i2i_shm.buf[:i2i_shm_size] = i2i_s  # 写入数据

    u2cart_wish_d = get_u2cart_wish(args.u2cart_wish_file)
    print('u2cart_wish_d size of mem [M]', asizeof.asizeof(u2cart_wish_d) / 1048576)  #
    u2cart_wish_s = pickle.dumps(u2cart_wish_d)
    u2cart_wish_shm_size = len(u2cart_wish_s)
    u2cart_wish_shm = shared_memory.SharedMemory(create=True, size=u2cart_wish_shm_size)
    u2cart_wish_shm.buf[:u2cart_wish_shm_size] = u2cart_wish_s  # 写入数据

    hot_i2leaf_d = get_hot_i2leaf(args.hot_i2leaf)
    print('hot_i2leaf_d size of mem [M]', asizeof.asizeof(hot_i2leaf_d) / 1048576)  #
    hot_i2leaf_s = pickle.dumps(hot_i2leaf_d)
    hot_i2leaf_shm_size = len(hot_i2leaf_s)
    hot_i2leaf_shm = shared_memory.SharedMemory(create=True, size=hot_i2leaf_shm_size)
    hot_i2leaf_shm.buf[:hot_i2leaf_shm_size] = hot_i2leaf_s  # 写入数据

    site_hot_d = get_site_hot(args.site_hot)
    print('site_hot_d size of mem [M]', asizeof.asizeof(site_hot_d) / 1048576)  #
    site_hot_s = pickle.dumps(site_hot_d)
    site_hot_shm_size = len(site_hot_s)
    site_hot_shm = shared_memory.SharedMemory(create=True, size=site_hot_shm_size)
    site_hot_shm.buf[:site_hot_shm_size] = site_hot_s  # 写入数据

    file_list = get_file_list(args)
    proc_list = [multiprocessing.Process(target=build_tfrecord, args=(
        fll[0], fll[1], fll[2],proc_id, args.stat_file,args.stat_flag, itm_shm.name, itm_shm_size, i2i_shm.name, i2i_shm_size, u2cart_wish_shm.name, u2cart_wish_shm_size,
        hot_i2leaf_shm.name, hot_i2leaf_shm_size, site_hot_shm.name, site_hot_shm_size,
        itm_stat_shm.name, itm_stat_shm_size
    )) for proc_id, fll in
                 enumerate(file_list)]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]
    fail_cnt = sum([p.exitcode for p in proc_list])
    itm_shm.close()
    itm_shm.unlink()
    itm_stat_shm.close()
    itm_stat_shm.unlink()
    i2i_shm.close()
    i2i_shm.unlink()
    u2cart_wish_shm.close()
    u2cart_wish_shm.unlink()
    hot_i2leaf_shm.close()
    hot_i2leaf_shm.unlink()
    site_hot_shm.close()
    site_hot_shm.unlink()
    if fail_cnt > 0:
        print('process fail cnt:', fail_cnt)
        # raise ValueError('Failed in %d process.' % fail_cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gentfr',
        description='gentfr',
        epilog='gentfr-help')
    parser.add_argument('--ds', type=str,
                        default=(datetime.today() - timedelta(days=1)).strftime('%Y%m%d'))
    parser.add_argument('--pre_ds', type=str,
                        default=(datetime.today() - timedelta(days=2)).strftime('%Y%m%d'))
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--range', type=str, default='')
    parser.add_argument('--thread', type=int, default=14)
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--dir_pt', default='cn_rec_detail_sample_v20_savana_in/ds=%s')
    parser.add_argument('--dir_tfr', default='cn_rec_detail_sample_v30_savana_in_tfr/ds=%s')
    parser.add_argument('--item_file', default='s3://warehouse-algo/rec/cn_rec_detail_feature_item_base/ds=%s/')
    parser.add_argument('--item_stat', default='s3://warehouse-algo/rec/features/cn_rec_detail_feature_item_stat/ds=%s/')
    parser.add_argument('--i2i_s3',
                        default='rec/recall/cn_rec_detail_recall_i2i_for_redis/item_user_debias_%s_1.0_0.7_0.5/')
    parser.add_argument('--i2i_file', default='swing_rec_Savana_IN_part_%s')
    parser.add_argument('--i2i_part', type=int, default=10)
    parser.add_argument('--u2cart_wish_file',
                        default='s3://warehouse-algo/rec/recall/cn_rec_detail_recall_wish_cart2i/ds=%s/')
    parser.add_argument('--hot_i2leaf', default='s3://warehouse-algo/rec/recall/cn_rec_detail_recall_main_leaf2i_ds/ds=%s/')
    parser.add_argument('--site_hot', default='s3://warehouse-algo/rec/recall/cn_rec_detail_recall_site_hot/ds=%s/')
    parser.add_argument('--stat_file', default='s3://warehouse-algo/rec/cn_rec_detail_sample_v30_savana_in_tfr_stat/ds=%s/')
    parser.add_argument('--stat_flag', type=bool, default=False)
    args = parser.parse_args()
    debug = args.debug
    if args.range != '':
        try:
            for ds in args.range.split(','):
                args = parser.parse_args() # 要重新初始化
                pre_ds = (datetime.strptime(ds, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
                st = time.time()
                args.ds = ds
                print('args.ds:', args.ds)
                args.pre_ds = pre_ds
                print('args.pre_ds:', args.pre_ds)
                args.item_file = args.item_file % args.ds
                args.item_stat = args.item_stat % pre_ds
                args.i2i_s3 = args.i2i_s3 % args.pre_ds
                args.u2cart_wish_file = args.u2cart_wish_file % pre_ds
                args.hot_i2leaf = args.hot_i2leaf % pre_ds
                args.site_hot = args.site_hot % pre_ds
                args.dir_pt = args.dir_pt % args.ds
                args.dir_tfr = args.dir_tfr % args.ds
                args.stat_file = args.stat_file % args.ds
                print('dir_pt', args.dir_pt)
                print('dir_tfr', args.dir_tfr)
                print('item_file', args.item_file)
                print('item_stat', args.item_stat)
                print('i2i_s3', args.i2i_s3)
                print('u2cart_wish_file', args.u2cart_wish_file)
                print('hot_i2leaf', args.hot_i2leaf)
                print('site_hot', args.site_hot)
                print('stat_file', args.stat_file)
                main(args)
                print('%s process %s cost %s' % (str(args.thread), ds, str(time.time() - st)))
        except Exception:
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)
            # print('data:',t)
    else:
        st = time.time()
        print('args.ds:', args.ds)
        pre_ds = (datetime.strptime(args.ds, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        args.pre_ds = pre_ds
        print('args.pre_ds:', args.pre_ds)
        args.item_file = args.item_file % args.ds
        args.item_stat = args.item_stat % pre_ds
        args.i2i_s3 = args.i2i_s3 % args.pre_ds
        args.u2cart_wish_file = args.u2cart_wish_file % pre_ds
        args.hot_i2leaf = args.hot_i2leaf % pre_ds
        args.site_hot = args.site_hot % pre_ds
        args.dir_pt = args.dir_pt % args.ds
        args.dir_tfr = args.dir_tfr % args.ds
        args.stat_file = args.stat_file % args.ds
        print('dir_pt', args.dir_pt)
        print('dir_tfr', args.dir_tfr)
        print('item_file', args.item_file)
        print('item_stat', args.item_stat)
        print('i2i_s3', args.i2i_s3)
        print('u2cart_wish_file', args.u2cart_wish_file)
        print('hot_i2leaf', args.hot_i2leaf)
        print('site_hot', args.site_hot)
        print('stat_file', args.stat_file)
        main(args)
        print('%s process %s cost %s' % (str(args.thread), args.ds, str(time.time() - st)))
    alert_feishu(f"gen tfr complete ds:{args.ds}")
# python gen_tfrecored_multi_process_with_gen_mt.py --ds=20250116  > run.log 2>&1 &  cpu:mem=9:7, 20250111
# python gen_tfrecored_multi_process_with_gen_mt.py --range=20250110,20250111,20250112,20250113,20250114,20250115,20250116,20250117,20250118,20250119  > run.log 2>&1 & done
# python gen_tfrecored_multi_process_with_gen_mt.py --range=20250101,20250102,20250103,20250104,20250105,20250106,20250107,20250108,20250109 > run.log 2>&1 & done
# python gen_tfrecored_multi_process_with_gen_mt.py --range=20241217,20241218,20241219,20241220,20241221,20241222,20241223,20241224> run.log 2>&1 & done
# python gen_tfrecored_multi_process_with_gen_mt.py --range=20241225,20241226,20241227,20241228,20241229,20241230,20241231,20250108,20250109 > run.log 2>&1 & done
