import argparse
import pyarrow as pa
from random import shuffle
from pyarrow import parquet
import tensorflow as tf
print(tf.__version__)
import tensorflow.compat.v1 as v1
import multiprocessing
import boto3
import os
import numpy as np
from sklearn.metrics import roc_auc_score

import math
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    c = math.ceil(len(lst) / n)
    for i in range(0, len(lst), c):
        yield lst[i:i + c]

def get_infer_tensor_dict(type=2):
    tensor_dict = {
        "cate_level1_id": tf.constant(["1"], dtype=tf.string),
        "cate_level2_id": tf.constant(["1"], dtype=tf.string),
        "cate_level3_id": tf.constant(["1"], dtype=tf.string),
        "cate_level4_id": tf.constant(["1"], dtype=tf.string),
        "country": tf.constant(["1"], dtype=tf.string),
        "ctr_7d": tf.constant([0.1], dtype=tf.float32),
        "cvr_7d": tf.constant([0.1], dtype=tf.float32),
        "show_7d": tf.constant([100], dtype=tf.int64),
        "click_7d": tf.constant([100], dtype=tf.int64),
        "cart_7d": tf.constant([100], dtype=tf.int64),
        "ord_total": tf.constant([100], dtype=tf.int64),
        "pay_total": tf.constant([100], dtype=tf.int64),
        "ord_7d": tf.constant([100], dtype=tf.int64),
        "pay_7d": tf.constant([100], dtype=tf.int64),
        "seq_goods_id": tf.constant(
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
             "19", "20"], dtype=tf.string),
        "goods_id": tf.constant(["1"], dtype=tf.string),
        "seq_cate_id": tf.constant(
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
             "19", "20"], dtype=tf.string),
        "cate_id": tf.constant(["1"], dtype=tf.string),

    },
    tensor_dict2 = {
        "cate_level1_id": tf.constant([["1"]], dtype=tf.string),
        "cate_level2_id": tf.constant([["1"]], dtype=tf.string),
        "cate_level3_id": tf.constant([["1"]], dtype=tf.string),
        "cate_level4_id": tf.constant([["1"]], dtype=tf.string),
        "country": tf.constant([["1"]], dtype=tf.string),
        "ctr_7d": tf.constant([[0.1]], dtype=tf.float32),
        "cvr_7d": tf.constant([[0.1]], dtype=tf.float32),
        "show_7d": tf.constant([[100]], dtype=tf.int64),
        "click_7d": tf.constant([[100]], dtype=tf.int64),
        "cart_7d": tf.constant([[100]], dtype=tf.int64),
        "ord_total": tf.constant([[100]], dtype=tf.int64),
        "pay_total": tf.constant([[100]], dtype=tf.int64),
        "ord_7d": tf.constant([[100]], dtype=tf.int64),
        "pay_7d": tf.constant([[100]], dtype=tf.int64),
        "seq_goods_id": tf.constant(
            [["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
             "19", "20"]], dtype=tf.string),
        "goods_id": tf.constant([["1"]], dtype=tf.string),
        "seq_cate_id": tf.constant(
            [["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
             "19", "20"]], dtype=tf.string),
        "cate_id": tf.constant([["1"]], dtype=tf.string),
        "sample_id": tf.constant([[2]], dtype=tf.int32),
    }
    tensor_dict3 = {
        "cate_level1_id": tf.constant([["1"],["1"]], dtype=tf.string),
        "cate_level2_id": tf.constant([["1"],["1"]], dtype=tf.string),
        "cate_level3_id": tf.constant([["1"],["1"]], dtype=tf.string),
        "cate_level4_id": tf.constant([["1"],["1"]], dtype=tf.string),
        "country": tf.constant([["1"],["1"]], dtype=tf.string),
        "ctr_7d": tf.constant([[0.1],[0.1]], dtype=tf.float32),
        "cvr_7d": tf.constant([[0.1],[0.1]], dtype=tf.float32),
        "show_7d": tf.constant([[100],[100]], dtype=tf.int64),
        "click_7d": tf.constant([[100],[100]], dtype=tf.int64),
        "cart_7d": tf.constant([[100],[100]], dtype=tf.int64),
        "ord_total": tf.constant([[100],[100]], dtype=tf.int64),
        "pay_total": tf.constant([[100],[100]], dtype=tf.int64),
        "ord_7d": tf.constant([[100],[100]], dtype=tf.int64),
        "pay_7d": tf.constant([[100],[100]], dtype=tf.int64),
        "seq_goods_id": tf.constant(
            [["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
              "19", "20"],["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
              "19", "20"]], dtype=tf.string),
        "goods_id": tf.constant([["1"],["1"]], dtype=tf.string),
        "seq_cate_id": tf.constant(
            [["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
              "19", "20"],["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
              "19", "20"]], dtype=tf.string),
        "cate_id": tf.constant([["1"],["1"]], dtype=tf.string),
        "sample_id": tf.constant([[2],[2]], dtype=tf.int32),
    }
    if type==1:
        return tensor_dict2
    elif type==2:
        return tensor_dict3
ID = 'sample_id'
SCORE = 'probabilities'
CLK = 'is_clk'
PAY = 'is_pay'

def process_tfr(tfr_list, batch_size, dir, score):

    def _parse_fea(data):
       feature_describe = {
           "ctr_7d": v1.FixedLenFeature(1, tf.float32, 0.0)
           , "cvr_7d": v1.FixedLenFeature(1, tf.float32, 0.0)
           , "show_7d": v1.FixedLenFeature(1, tf.int64, 0)
           , "click_7d": v1.FixedLenFeature(1, tf.int64, 0)
           , "cart_7d": v1.FixedLenFeature(1, tf.int64, 0)
           , "ord_total": v1.FixedLenFeature(1, tf.int64, 0)
           , "pay_total": v1.FixedLenFeature(1, tf.int64, 0)
           , "ord_7d": v1.FixedLenFeature(1, tf.int64, 0)
           , "pay_7d": v1.FixedLenFeature(1, tf.int64, 0)

           , "cate_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "goods_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "cate_level1_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "cate_level2_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "cate_level3_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "cate_level4_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "country": v1.FixedLenFeature(1, tf.string, '-1')

           # , "seq_cate_id": v1.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
           # , "seq_goods_id": v1.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
           , "seq_cate_id": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
           , "seq_goods_id": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)

           , "is_clk": v1.FixedLenFeature(1, tf.int64, 0)
           , "is_pay": v1.FixedLenFeature(1, tf.int64, 0)
           , "sample_id": v1.FixedLenFeature(1, tf.string, "-1")
       }
       features = tf.io.parse_single_example(data, features=feature_describe)
       return features
    os.system('mkdir tmp')
    for file in tfr_list:
        print('download file into tmp:',file)
        os.system('aws s3 cp %s %s' % (file, './tmp/'))
    file_suffix = ['./tmp/' + e.split('/')[-1] for e in tfr_list]
    ds = tf.data.TFRecordDataset(file_suffix)
    ds = ds.map(_parse_fea).batch(batch_size)
    item_features_string = {"goods_id": "", "cate_id": "", "cate_level1_id": "", "cate_level2_id": "",
                            "cate_level3_id": "", "cate_level4_id": "", "country": ""}
    item_features_double = {"ctr_7d": 0.0, "cvr_7d": 0.0}
    item_features_int = {"show_7d": 0, "click_7d": 0, "cart_7d": 0, "ord_total": 0, "pay_total": 0, "ord_7d": 0,
                         "pay_7d": 0}
    user_seq_string = {"seq_goods_id": [""] * 20, "seq_cate_id": [""] * 20}
    predictor = tf.saved_model.load(dir).signatures["serving_default"]
    for idx in ds.as_numpy_iterator():
        feed_dict = {}
        id = idx[ID].tolist()
        id = [e[0] for e in id]
        if ID not in score:
            score[ID] = id
        else:
            score[ID].extend(id)
        
        # is_clk
        clk = idx[CLK].tolist()
        clk = [e[0] for e in clk]
        if CLK not in score:
            score[CLK] = clk
        else:
            score[CLK].extend(clk)
        # is_pay
        pay = idx[PAY].tolist()
        pay = [e[0] for e in pay]
        if PAY not in score:
            score[PAY] = pay
        else:
            score[PAY].extend(pay)

        for name in item_features_string.keys():
            feed_dict[name] = tf.constant(idx[name], dtype=tf.string)
        for name in item_features_int.keys():
            feed_dict[name] = tf.constant(idx[name], dtype=tf.int64)
        for name in item_features_double.keys():
            feed_dict[name] = tf.constant(idx[name], dtype=tf.float32)
        for name in user_seq_string.keys():
            feed_dict[name] = tf.constant(idx[name], dtype=tf.string)
        res = predictor(**feed_dict)
        prob = res[SCORE].numpy().tolist()
        prob = [e[0] for e in prob]
        if SCORE not in score:
            score[SCORE] = prob
        else:
            score[SCORE].extend(prob)
        # print('res', res)

def main(args):
    s3_cli = boto3.client('s3')
    BUCKET = 'warehouse-algo'

    # get files
    paginator = s3_cli.get_paginator('list_objects_v2')
    page_iter = paginator.paginate(Bucket=BUCKET, Prefix=args.tfr_s3)
    file_list = [[v['Key'] for v in page.get('Contents', [])] for page in page_iter][0]
    file_list = ['s3://%s/%s'%(BUCKET, v) for v in file_list]
    print('file list in dir', file_list)
    shuffle(file_list)
    file_batch = list(chunks(file_list,  args.proc))

    manager = multiprocessing.Manager()
    score = manager.dict()
    # score[ID] = []
    # score[SCORE] = []
    jobs = []
    for tfr_list in file_batch:
        p = multiprocessing.Process(target=process_tfr, args=(tfr_list, args.batch_size, args.dir, score))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    score = dict(score)
    tb = pa.table(score)
    parquet.write_table(tb, args.tb)
    y_score = score[SCORE]
    is_clk = score[CLK]
    avg_pred = np.mean(y_score)
    avg_label = np.mean(is_clk)
    print('avg_pred:', avg_pred, 'avg_label:', avg_label)
    auc = roc_auc_score(is_clk, y_score)
    print('auc:', auc, 'avg_pred:', avg_pred, 'avg_label:', avg_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='predict',
        description='predict',
        epilog='predict')
    parser.add_argument('--tfr', default='./part-00000-1186234f-fa44-44a8-9aff-08bcf2c5fb26-c000')
    parser.add_argument('--tfr_s3', default='rec/cn_rec_detail_sample_v1_tfr_cvr/ds=20241113/')
    parser.add_argument('--tb', default='s3://warehouse-algo/rec/model_pred/predict_v3')
    parser.add_argument('--dir', default='/home/sagemaker-user/mayfair/algo_rec/rank/prod/model_tmp')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--proc', type=int, default=1)
    args = parser.parse_args()
    main(args)
