# encoding:utf-8
import tensorflow as tf
import sys
import time
import multiprocessing
sys.path.append('/home/sagemaker-user/mayfair')
from algo_rec.constant import *
from pyarrow import parquet
import argparse
import boto3
import math
import numpy as np

s3_cli = boto3.client('s3')
BUCKET = 'warehouse-algo'


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

def build_tfrecord(from_file_list, out_file_list):
    for from_file, out_file in zip(from_file_list, out_file_list):
        st = time.time()
        pt = parquet.read_table(from_file)
        ed = time.time()
        print('read ptpath:%s data cost:%s' % (from_file, str(ed - st)))
        st = time.time()
        fout = tf.python_io.TFRecordWriter(out_file)
        for t in zip(
                pt["ctr_7d"], pt["cvr_7d"]
                , pt["cate_id"], pt["goods_id"], pt["cate_level1_id"], pt["cate_level2_id"], pt["cate_level3_id"],
                pt["cate_level4_id"], pt["country"]
                , pt["show_7d"], pt["click_7d"], pt["cart_7d"], pt["ord_total"], pt["pay_total"], pt["ord_7d"],
                pt["pay_7d"], pt["is_clk"], pt["is_pay"]
                , pt["seq_cate_id"], pt["seq_goods_id"]
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
            sample = tf.train.Example(features=tf.train.Features(feature=feature))
            record = sample.SerializeToString()
            fout.write(record)
        ed = time.time()
        print('gen trf done, cost %s' % str(ed - st))

def split_list_into_batch(data_list, batch_count=None, batch_size=None):
    assert data_list and (batch_count or batch_size)
    batch_size = batch_size or math.ceil(len(data_list) / batch_count)
    batch_count = math.ceil(len(data_list) / batch_size)
    for idx in range(batch_count):
        yield data_list[idx * batch_size: (idx + 1) * batch_size]

def run_multi_process(func, ds, batch):
    trf_path_local = './cn_rec_detail_sample_v1_test' + ds
    ptpath = s3_sp_pt_dir + ds
    tfr_path_s3 = s3_sp_tfr_dir + ds + 'test'
    # get files
    paginator = s3_cli.get_paginator('list_objects_v2')
    page_iter = paginator.paginate(Bucket=BUCKET, Prefix=s3_sp_pt_dir_key + ds)
    file_list = [[v['Key'] for v in page.get('Contents', [])] for page in page_iter][0]
    # batch
    file_batch = np.array_split(file_list, batch)
    args_list = []
    for ll in file_batch:
        tmp = []
        for file in ll:
            tmp.append((ptpath + file, trf_path_local + file))
        args_list.append(tmp)
    print('args_list:', args_list)
    # multiprocess
    proc_list = [multiprocessing.Process(target=func, args=args) for args in args_list]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]
    fail_cnt = sum([p.exitcode for p in proc_list])
    if fail_cnt:
        raise ValueError('Failed in %d process.' % fail_cnt)

def main(args):
    run_multi_process(build_tfrecord, args.ds, 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gentfr',
        description='gentfr',
        epilog='gentfr-help')
    parser.add_argument('--ds', default='ds=20241113')
    args = parser.parse_args()
    main(args)
