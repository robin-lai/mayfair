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
CTR = 'ctr'
CVR = 'cvr'
CTCVR = 'ctcvr'
CLK = 'is_clk'
PAY = 'is_pay'
prod_model = 's3://warehouse-algo/rec/prod_model/'
tmp_dir = '/home/sagemaker-user/tmp/'
tmp_dir_data = tmp_dir + 'data/'

def process_tfr(thread_idx, tfr_list, batch_size, dir, score):
    score[thread_idx] = {}

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
           , "highLevelSeqListGoods": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
           , "highLevelSeqListCateId": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
           , "lowerLevelSeqListGoods": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
           , "lowerLevelSeqListCateId": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)

           , "is_clk": v1.FixedLenFeature(1, tf.int64, 0)
           , "is_pay": v1.FixedLenFeature(1, tf.int64, 0)
           , "sample_id": v1.FixedLenFeature(1, tf.string, "-1")
       }
       features = tf.io.parse_single_example(data, features=feature_describe)
       return features
    os.system('mkdir -p %s'%tmp_dir_data)
    for file in tfr_list:
        print('download file into tmp:',file)
        os.system('aws s3 cp %s %s' % (file, tmp_dir_data))
    file_suffix = [tmp_dir_data + e.split('/')[-1] for e in tfr_list]
    ds = tf.data.TFRecordDataset(file_suffix)
    ds = ds.map(_parse_fea).batch(batch_size)
    item_features_string = {"goods_id": "", "cate_id": "", "cate_level1_id": "", "cate_level2_id": "",
                            "cate_level3_id": "", "cate_level4_id": "", "country": ""}
    item_features_double = {"ctr_7d": 0.0, "cvr_7d": 0.0}
    item_features_int = {"show_7d": 0, "click_7d": 0, "cart_7d": 0, "ord_total": 0, "pay_total": 0, "ord_7d": 0,
                         "pay_7d": 0}
    user_seq_string = {"seq_goods_id": [""] * 20, "seq_cate_id": [""] * 20, "highLevelSeqListGoods": [""] * 20,
                       "highLevelSeqListCateId": [""] * 20, "lowerLevelSeqListGoods": [""] * 20,
                       "lowerLevelSeqListCateId": [""] * 20
                       }
    predictor = tf.saved_model.load(dir).signatures["serving_default"]
    for idx in ds.as_numpy_iterator():
        feed_dict = {}
        id = idx[ID].tolist()
        id = [e[0] for e in id]
        if ID not in score[thread_idx]:
            score[thread_idx][ID] = id
        else:
            score[thread_idx][ID].extend(id)
        
        # is_clk
        clk = idx[CLK].tolist()
        clk = [e[0] for e in clk]
        if CLK not in score[thread_idx]:
            score[thread_idx][CLK] = clk
        else:
            score[thread_idx][CLK].extend(clk)
        # is_pay
        pay = idx[PAY].tolist()
        pay = [e[0] for e in pay]
        if PAY not in score[thread_idx]:
            score[thread_idx][PAY] = pay
        else:
            score[thread_idx][PAY].extend(pay)

        for name in item_features_string.keys():
            feed_dict[name] = tf.constant(idx[name], dtype=tf.string)
        for name in item_features_int.keys():
            feed_dict[name] = tf.constant(idx[name], dtype=tf.int64)
        for name in item_features_double.keys():
            feed_dict[name] = tf.constant(idx[name], dtype=tf.float32)
        for name in user_seq_string.keys():
            feed_dict[name] = tf.constant(idx[name], dtype=tf.string)
        res = predictor(**feed_dict)
        
        prob = res[CTR].numpy().tolist()
        prob = [e[0] for e in prob]
        if CTR not in score[thread_idx]:
            score[thread_idx][CTR] = prob
        else:
            score[thread_idx][CTR].extend(prob)
        # print('res', res)
        cvr = res[CVR].numpy().tolist()
        cvr = [e[0] for e in cvr]
        if CVR not in score[thread_idx]:
            score[thread_idx][CVR] = cvr
        else:
            score[thread_idx][CVR].extend(cvr)
        # ctcvr
        ctcvr = res[CTCVR].numpy().tolist()
        ctcvr = [e[0] for e in ctcvr]
        if CTCVR not in score[thread_idx]:
            score[thread_idx][CTCVR] = ctcvr
        else:
            score[thread_idx][CTCVR].extend(ctcvr)

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

    # download model
    s3_model = prod_model + args.model_name + args.model_version
    model_local = tmp_dir + args.model_name + args.model_version
    os.system("rm -rf %s" % (model_local))
    os.system("mkdir -p %s" % (model_local))
    os.system("aws s3 cp --recursive %s %s" % (s3_model, model_local))

    manager = multiprocessing.Manager()
    score = manager.dict()
    # score[ID] = []
    # score[SCORE] = []
    jobs = []
    for thread_idx, tfr_list in enumerate(file_batch):
        p = multiprocessing.Process(target=process_tfr, args=(thread_idx, tfr_list[0], args.batch_size, model_local, score))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    # merge multi thread score
    score = dict(score)
    merge_score = dict()
    for thread_i, s in score.items():
        for k, v in s.items():
            if k in merge_score:
                merge_score[k].extend(v)
            else:
                merge_score[k] = v

    # save
    tb = pa.table(merge_score)
    parquet.write_table(tb, args.tb)

    # auc
    pctr = merge_score[CTR]
    is_clk = merge_score[CLK]
    avg_pred_ctr = np.mean(pctr)
    avg_label_clk = np.mean(is_clk)
    print('N:',len(pctr), 'avg_pred_ctr:', avg_pred_ctr, 'avg_label_clk:', avg_label_clk)
    auc = roc_auc_score(is_clk, pctr)
    print('ctr-auc:', auc)

    pcvr, is_pay = [], []
    for e1, e2, e3 in zip(merge_score[CLK], merge_score[PAY], merge_score[CVR]):
        if e1 == 1:
            is_pay.append(e2)
            pcvr.append(e3)
    avg_pred_cvr = np.mean(pcvr)
    avg_label_pay = np.mean(is_pay)
    print('N:',len(pcvr), 'avg_pred_cvr:', avg_pred_cvr, 'avg_label_pay:', avg_label_pay)
    auc = roc_auc_score(is_pay, pcvr)
    print('cvr-auc:', auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='predict',
        description='predict',
        epilog='predict')
    parser.add_argument('--model_name', default='prod_mtl_seq_all_esmm_v0')
    parser.add_argument('--model_version', default='/ds=20241205/model/1733652742')
    parser.add_argument('--tfr', default='./part-00000-1186234f-fa44-44a8-9aff-08bcf2c5fb26-c000')
    parser.add_argument('--tfr_s3', default='rec/cn_rec_detail_sample_v10_ctr/ds=20241206/')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--proc', type=int, default=1)
    args = parser.parse_args()
    main(args)
