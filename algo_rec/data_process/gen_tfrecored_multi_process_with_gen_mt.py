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
import datetime
from pympler import asizeof

import math


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


def build_tfrecord(path_pt_list, path_tfr_local_list, path_tfr_s3_list, shm_i2i_name, shm_i2i_size, shm_itm_name,
                   shm_itm_size):
    shm_i2i = shared_memory.SharedMemory(name=shm_i2i_name)
    sd_i2i = bytes(shm_i2i.buf[:shm_i2i_size])
    i2i_d = pickle.loads(sd_i2i)  # 反序列化为嵌套字典
    shm_itm = shared_memory.SharedMemory(name=shm_itm_name)
    sd_itm = bytes(shm_itm.buf[:shm_itm_size])
    itm_d = pickle.loads(sd_itm)  # 反序列化为嵌套字典

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

    def build_mt(tt, feature):
        mt = []
        mt_w = []
        main_goods = int(tt['main_goods_id'])
        tgt_id = int(tt['goods_id'])
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
        if len(mt) > 1:
            print('feature', feature)
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
        item_features_double = {"ctr_7d": 0.0, "cvr_7d": 0.0}
        item_features_int = {"show_7d": 0, "click_7d": 0, "cart_7d": 0, "ord_total": 0, "pay_total": 0, "ord_7d": 0,
                             "pay_7d": 0,
                             "is_rel_cate": 0, "is_rel_cate2": 0, "is_rel_cate3": 0, "is_rel_cate4": 0, "sales_price": 0
                             }
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
        for name in item_features_double.keys():
            feature.update({name: floats_fea(t[name])})
        for name in item_features_int.keys():
            feature.update({name: ints_fea(t[name])})
        for name in user_int.keys():
            feature.update({name: ints_fea(t[name])})
        for name in other_int.keys():
            feature.update({name: ints_fea(t[name])})
        for name in user_seq_string.keys():
            feature.update({name: bytes_fea(t[name], n=20)})

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
            build_mt(t, feature)
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


def get_file_list(args):
    s3_cli = boto3.client('s3')
    BUCKET = 'warehouse-algo'
    s3_buk = "s3://warehouse-algo/"
    s3_obj = "rec/"
    local_data = "/home/sagemaker-user/mayfair/algo_rec/data/"
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
    return args_list


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
                tmp_d = {}
                for tt in v.split(chr(2)):
                    tokens = tt.split(chr(4))
                    tmp_d[int(tokens[0])] = tokens[1]
                i2i_d[int(k)] = tmp_d
    print('read i2i end, num:', len(i2i_d.keys()))
    return i2i_d


def get_item_feature(file):
    ret = {}
    pt = parquet.read_table(file).to_pylist()
    for e in pt:
        ret[int(e['goods_id'])] = e
    return ret


def main(args):
    itm_d = get_item_feature(args.item % args.ds)
    print('item_feature size of mem [M]', asizeof.asizeof(itm_d) / 1048576)  #
    print('get item features:', len(itm_d.keys()))
    i2i_d = get_i2i(args.i2i_part, args.i2i_s3, args.i2i_file)
    print('i2i_d size of mem [M]', asizeof.asizeof(i2i_d) / 1048576)  #
    # run_multi_process(build_tfrecord, args, i2i_d, item_feature)

    serialized_data = pickle.dumps(i2i_d)
    shm_size = len(serialized_data)
    # 创建共享内存并写入序列化数据
    shm = shared_memory.SharedMemory(create=True, size=shm_size)
    shm.buf[:shm_size] = serialized_data  # 写入数据
    print(f"Shared memory created. Name: {shm.name}, Size: {shm_size} bytes")

    serialized_data_itm = pickle.dumps(itm_d)
    shm_itm_size = len(serialized_data_itm)
    shm_itm = shared_memory.SharedMemory(create=True, size=shm_itm_size)
    shm_itm.buf[:shm_itm_size] = serialized_data_itm  # 写入数据
    print(f"Shared memory created. Name: {shm_itm.name}, Size: {shm_itm_size} bytes")

    args_list = get_file_list(args)
    proc_list = [multiprocessing.Process(target=build_tfrecord, args=(
        args[0], args[1], args[2], shm.name, shm_size, shm_itm.name, shm_itm_size)) for args in args_list]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]
    fail_cnt = sum([p.exitcode for p in proc_list])
    shm.close()
    shm.unlink()
    shm_itm.close()
    shm_itm.unlink()
    if fail_cnt > 0:
        print('process fail cnt:', fail_cnt)
        # raise ValueError('Failed in %d process.' % fail_cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gentfr',
        description='gentfr',
        epilog='gentfr-help')
    parser.add_argument('--ds', type=str,
                        default='ds=%s' % (datetime.date.today() - datetime.timedelta(days=2)).strftime('%Y%m%d'))
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--range', type=str, default='')
    parser.add_argument('--thread', type=int, default=15)
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--dir_pt', default='cn_rec_detail_sample_v20_savana_in/')
    parser.add_argument('--dir_tfr', default='cn_rec_detail_sample_v20_savana_in_tfr/')
    parser.add_argument('--item', default='s3://warehouse-algo/rec/cn_rec_detail_feature_item_base/%s/')
    parser.add_argument('--i2i_s3',
                        default='s3://warehouse-algo/rec/recall/cn_rec_detail_recall_i2i_for_redis%s/item_user_debias_%s/')
    parser.add_argument('--i2i_file', default='swing_rec_Savana_IN_part_%s')
    parser.add_argument('--i2i_part', type=int, default=7)
    parser.add_argument('--v', default='')

    args = parser.parse_args()
    debug = args.debug
    if args.range != '':
        try:
            for ds in args.range.split(','):
                st = time.time()
                args.ds = 'ds=' + ds
                print('args.ds:', args.ds)
                # item_feature = get_item_feature(args.item % args.ds)
                args.i2i_s3 = args.i2i_s3 % (args.v, args.ds)
                main(args)
                print('%s process %s cost %s' % (str(args.thread), ds, str(time.time() - st)))
        except Exception:
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)
            # print('data:',t)
    else:
        st = time.time()
        args.i2i_s3 = args.i2i_s3 % (args.v, args.ds[3:])
        print('args.ds:', args.ds)
        main(args)
        print('%s process %s cost %s' % (str(args.thread), args.ds, str(time.time() - st)))
