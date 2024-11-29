# encoding:utf-8
import os

import tensorflow as tf
import sys
import time
import multiprocessing
from pyarrow import parquet
import argparse
import boto3
from random import shuffle
import numpy as np

s3_cli = boto3.client('s3')
BUCKET = 'warehouse-algo'

s3_sp_pt_dir = "s3://warehouse-algo/rec/cn_rec_detail_sample_v1/"
s3_sp_pt_dir_key = "rec/cn_rec_detail_sample_v1/"

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
        v_list.extend([" "] * (n - len(v_list)))
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

def build_tfrecord(*args):
    from_file_list = args[0]
    out_file_ctr_list = args[1]
    out_file_cvr_list = args[2]
    for from_file, out_file_ctr_file, out_file_cvr_file in zip(from_file_list, out_file_ctr_list, out_file_cvr_list):
        st = time.time()
        pt = parquet.read_table(from_file)
        ed = time.time()
        print('read ptpath:%s data cost:%s' % (from_file, str(ed - st)))
        st = time.time()
        fout_cvr = tf.python_io.TFRecordWriter(out_file_cvr_file)
        fout_ctr = tf.python_io.TFRecordWriter(out_file_ctr_file)
        for t in zip(
                pt["ctr_7d"], pt["cvr_7d"]
                , pt["cate_id"], pt["goods_id"], pt["cate_level1_id"], pt["cate_level2_id"], pt["cate_level3_id"],
                pt["cate_level4_id"], pt["country"]
                , pt["show_7d"], pt["click_7d"], pt["cart_7d"], pt["ord_total"], pt["pay_total"], pt["ord_7d"],
                pt["pay_7d"], pt["is_clk"], pt["is_pay"]
                , pt["seq_cate_id"], pt["seq_goods_id"]
                , pt["sample_id"]
        ):
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
            sample = tf.train.Example(features=tf.train.Features(feature=feature))
            record = sample.SerializeToString()
            fout_ctr.write(record)
            if t[16].as_py() == 1:
                fout_cvr.write(record)
        ed = time.time()
        print('gen trf done, cost %s' % str(ed - st))
        # upload
        # print('upload from %s to %s' % (out_file_cvr_file, out_file_ctr_file))
        # os.system('aws s3 cp %s %s' % (out_file_cvr_file, out_file_ctr_file))


def split_list_into_batch(data_list, batch_count=None, batch_size=None):
    assert data_list and (batch_count or batch_size)
    batch_size = batch_size or math.ceil(len(data_list) / batch_count)
    batch_count = math.ceil(len(data_list) / batch_size)
    for idx in range(batch_count):
        yield data_list[idx * batch_size: (idx + 1) * batch_size]

def run_multi_process(func,args):
    trf_path_local_ctr = args.dir_ctr + args.ds
    ptpath = s3_sp_pt_dir + args.ds
    trf_path_local_cvr = args.dir_cvr + args.ds
    os.system('mkdir -p %s' % trf_path_local_ctr)
    os.system('mkdir -p %s' % trf_path_local_cvr)
    # get files
    paginator = s3_cli.get_paginator('list_objects_v2')
    print('key:', s3_sp_pt_dir_key + args.ds)
    page_iter = paginator.paginate(Bucket=BUCKET, Prefix=s3_sp_pt_dir_key + args.ds)
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
        pt_path_tmp = []
        local_path_tmp_ctr = []
        local_path_tmp_cvr = []
        for file in ll:
            pt_path_tmp.append(ptpath +  '/' + file)
            local_path_tmp_ctr.append(trf_path_local_cvr + '/' + file)
            local_path_tmp_cvr.append(trf_path_local_ctr + '/' + file)

        args_list.append([pt_path_tmp, local_path_tmp_ctr, local_path_tmp_cvr])
    print('args_list top ', args_list)
    # multiprocess
    proc_list = [multiprocessing.Process(target=func, args=args) for args in args_list]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]
    fail_cnt = sum([p.exitcode for p in proc_list])
    if fail_cnt:
        raise ValueError('Failed in %d process.' % fail_cnt)

def main(args):
    run_multi_process(build_tfrecord, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gentfr',
        description='gentfr',
        epilog='gentfr-help')
    parser.add_argument('--ds', default='ds=20241113')
    parser.add_argument('--range', type=str, default='20241102')
    parser.add_argument('--thread', type=int, default=20)
    parser.add_argument('--s3', type=bool, default=False)
    parser.add_argument('--dir_ctr', default='~/mayfair/algo_rec/data/cn_rec_detail_sample_v1_ctr/')
    parser.add_argument('--dir_cvr', default='~/mayfair/algo_rec/data/cn_rec_detail_sample_v1_cvr/')
    args = parser.parse_args()
    for ds in args.range.split(','):
        st = time.time()
        args.ds = 'ds=' + ds
        main(args)
        print('%s process %s cost %s' % (str(args.thread), ds, str(time.time() - st)))

