# encoding:utf-8
import os
import json
import tensorflow as tf
import sys
import time
import multiprocessing
from pyarrow import parquet
import argparse
import boto3
from random import shuffle
import numpy as np
from statsmodels.sandbox.distributions.examples.matchdist import categ

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

def build_tfrecord(path_pt_list, path_tfr_local_list, path_tfr_s3_list):
    def build_seq_on(seq_on):
        if debug:
            print('seq_on',seq_on)
        ret = {}
        if seq_on == "":
            js = dict()
        else:
            js = dict(json.loads(seq_on))
        for k, v in js.items():
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
                cate_id_list.append(str(item_feature[e]["cate_id"]))
                cate_name_list.append(str(item_feature[e]["cate_name"]))
                cate_level3_id_list.append(str(item_feature[e]["cate_level3_id"]))
                cate_level3_name_list.append(str(item_feature[e]["cate_level3_name"]))
                cate_level4_id_list.append(str(item_feature[e]["cate_level4_id"]))
                cate_level4_name_list.append(str(item_feature[e]["cate_level4_name"]))
            n = 20 if len(goods) >= 20 else len(goods)
            ret[k + 'Goods'] = bytes_fea(goods, n=20)
            ret[k + 'CateId'] = bytes_fea(cate_id_list, n=20)
            ret[k + 'CateName'] = bytes_fea(cate_name_list, n=20)
            ret[k + 'CateId3'] = bytes_fea(cate_level3_id_list, n=20)
            ret[k + 'CateName3'] = bytes_fea(cate_level3_name_list, n=20)
            ret[k + 'CateId4'] = bytes_fea(cate_level4_id_list, n=20)
            ret[k + 'CateName4'] = bytes_fea(cate_level4_name_list, n=20)
            ret[k + '_len'] = ints_fea(n)
        return ret

    def build_feature(t):
        feature = dict()
        feature.update({"ctr_7d": floats_fea(t[0].as_py())})
        feature.update({"cvr_7d": floats_fea(t[1].as_py())})
        feature.update({"cate_id": bytes_fea(t[2].as_py())})
        feature.update({"goods_id": bytes_fea(t[3].as_py())})
        feature.update({"cate_level1_id": bytes_fea(t[4].as_py())})
        feature.update({"cate_level2_id": bytes_fea(t[5].as_py())})
        feature.update({"cate_level3_id": bytes_fea(t[6].as_py())})
        feature.update({"cate_level4_id": bytes_fea(t[7].as_py())})
        feature.update({"country": bytes_fea(t[8].as_py())})
        feature.update({"show_7d": ints_fea(t[9].as_py())})
        feature.update({"click_7d": ints_fea(t[10].as_py())})
        feature.update({"cart_7d": ints_fea(t[11].as_py())})
        feature.update({"ord_total": ints_fea(t[12].as_py())})
        feature.update({"pay_total": ints_fea(t[13].as_py())})
        feature.update({"ord_7d": ints_fea(t[14].as_py())})
        feature.update({"pay_7d": ints_fea(t[15].as_py())})
        feature.update({"is_clk": ints_fea(t[16].as_py())})
        feature.update({"is_pay": ints_fea(t[17].as_py())})
        feature.update({"seq_cate_id": bytes_fea(t[18].as_py(), n=20)})
        feature.update({"seq_goods_id": bytes_fea(t[19].as_py(), n=20)})
        feature.update({"sample_id": bytes_fea(t[20].as_py())})
        feature.update(build_seq_on(t[21].as_py()))
        return feature

    for pt_file, tfr_local_file, tfr_s3_file in zip(path_pt_list, path_tfr_local_list, path_tfr_s3_list):
        st = time.time()
        pt = parquet.read_table(pt_file)
        ed = time.time()
        print('read ptpath:%s data cost:%s' % (pt_file, str(ed - st)))
        st = time.time()
        fout_ctr = tf.io.TFRecordWriter(tfr_local_file)
        for t in zip(
                pt["ctr_7d"], pt["cvr_7d"]
                , pt["cate_id"], pt["goods_id"], pt["cate_level1_id"], pt["cate_level2_id"], pt["cate_level3_id"],
                pt["cate_level4_id"], pt["country"]
                , pt["show_7d"], pt["click_7d"], pt["cart_7d"], pt["ord_total"], pt["pay_total"], pt["ord_7d"],
                pt["pay_7d"], pt["is_clk"], pt["is_pay"]
                , pt["seq_cate_id"], pt["seq_goods_id"]
                , pt["sample_id"], pt['seq_on']
        ):
            feature = dict()
            try:
                feature = build_feature(t)
                sample = tf.train.Example(features=tf.train.Features(feature=feature))
                record = sample.SerializeToString()
                fout_ctr.write(record)
            except Exception as e:
                print(e)
                print('data:',t)
            if debug:
                print('features',feature)
        ed = time.time()
        fout_ctr.close()
        print('gen trf done, cost %s' % str(ed - st))
        # upload
        print('upload from %s to %s' % (tfr_local_file, tfr_s3_file))
        os.system('aws s3 cp %s %s' % (tfr_local_file, tfr_s3_file))


def split_list_into_batch(data_list, batch_count=None, batch_size=None):
    assert data_list and (batch_count or batch_size)
    batch_size = batch_size or math.ceil(len(data_list) / batch_count)
    batch_count = math.ceil(len(data_list) / batch_size)
    for idx in range(batch_count):
        yield data_list[idx * batch_size: (idx + 1) * batch_size]

def run_multi_process(func,args):
    path_pt = s3_buk + s3_obj + args.dir_pt + args.ds
    path_pt_suffix = s3_obj + args.dir_pt + args.ds
    path_tfr_local = local_data + args.dir_tfr + args.ds
    path_tfr_local_base = local_data + args.dir_tfr
    path_tfr_s3 = s3_buk + s3_obj + args.dir_tfr + args.ds
    os.system('rm %s' % path_tfr_local_base)
    os.system('mkdir -p %s' % path_tfr_local)
    # get files
    paginator = s3_cli.get_paginator('list_objects_v2')
    print('key:', path_pt_suffix)
    page_iter = paginator.paginate(Bucket=BUCKET, Prefix=path_pt_suffix)
    file_list = [[v['Key'] for v in page.get('Contents', [])] for page in page_iter][0]
    file_suffix_list = [v.split('/')[-1] for v in file_list]
    print('file list in dir', file_list)
    print('file_suffix_list:', file_suffix_list)
    # batch
    shuffle(file_suffix_list)
    file_batch = list(chunks(file_suffix_list,  args.thread))
    # file_batch = np.array_split(file_suffix_list, args.thread)
    args_list = []
    for ll in file_batch:
        l1, l2, l3 = [],[],[]
        for file in ll:
            l1.append(path_pt +  '/' + file)
            l2.append(path_tfr_local + '/' + file)
            l3.append(path_tfr_s3 +  '/' + file)
        args_list.append([l1, l2, l3])
    print('args_list top ', args_list)
    # multiprocess
    proc_list = [multiprocessing.Process(target=func, args=(args[0], args[1], args[2])) for args in args_list]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]
    fail_cnt = sum([p.exitcode for p in proc_list])
    if fail_cnt:
        raise ValueError('Failed in %d process.' % fail_cnt)

def get_item_feature(file):
    ret = {}
    pt = parquet.read_table(file).to_pylist()
    for e in pt:
        ret[int(e['goods_id'])] = e
    return ret


def main(args):
    run_multi_process(build_tfrecord, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gentfr',
        description='gentfr',
        epilog='gentfr-help')
    parser.add_argument('--ds', default='ds=20241202')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--range', type=str, default='')
    parser.add_argument('--thread', type=int, default=15)
    parser.add_argument('--dir_pt', default='cn_rec_detail_sample_v10/')
    parser.add_argument('--dir_tfr', default='cn_rec_detail_sample_v10_ctr/')
    parser.add_argument('--item', default='s3://algo-sg/rec/cn_rec_detail_feature_item_base/ds=20241206/')

    args = parser.parse_args()
    item_feature = get_item_feature(args.item)
    debug = args.debug
    print('get item features:', len(item_feature.keys()))
    if args.range != '':
        for ds in args.range.split(','):
            st = time.time()
            args.ds = 'ds=' + ds
            print('args.ds:', args.ds)
            main(args)
            print('%s process %s cost %s' % (str(args.thread), ds, str(time.time() - st)))
    else:
        st = time.time()
        print('args.ds:', args.ds)
        main(args)
        print('%s process %s cost %s' % (str(args.thread), args.ds, str(time.time() - st)))


