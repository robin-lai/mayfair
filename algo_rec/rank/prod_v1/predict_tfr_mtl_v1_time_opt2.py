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
import time
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
pred_dir = 's3://warehouse-algo/rec/model_pred/'
tmp_dir = '/home/sagemaker-user/tmp/'
tmp_dir_data = tmp_dir + 'data/'

debug = False

def process_tfr(proc, tfr_list, batch_size, dir, score):

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

           , "is_rel_cate": v1.FixedLenFeature(1, tf.int64, 0)
           , "is_rel_cate2": v1.FixedLenFeature(1, tf.int64, 0)
           , "is_rel_cate3": v1.FixedLenFeature(1, tf.int64, 0)
           , "is_rel_cate4": v1.FixedLenFeature(1, tf.int64, 0)
           , "sales_price": v1.FixedLenFeature(1, tf.int64, 0)

           , "main_goods_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "main_cate_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "main_cate_level2_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "main_cate_level3_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "main_cate_level4_id": v1.FixedLenFeature(1, tf.string, "-1")

           , "prop_seaon": v1.FixedLenFeature(1, tf.string, "-1")
           , "prop_length": v1.FixedLenFeature(1, tf.string, "-1")
           , "prop_main_material": v1.FixedLenFeature(1, tf.string, "-1")
           , "prop_pattern": v1.FixedLenFeature(1, tf.string, "-1")
           , "prop_style": v1.FixedLenFeature(1, tf.string, "-1")
           , "prop_quantity": v1.FixedLenFeature(1, tf.string, "-1")
           , "prop_fitness": v1.FixedLenFeature(1, tf.string, "-1")

           , "last_login_device": v1.FixedLenFeature(1, tf.string, "-1")
           , "last_login_brand": v1.FixedLenFeature(1, tf.string, "-1")
           , "register_brand": v1.FixedLenFeature(1, tf.string, "-1")

           , "cate_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "goods_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "cate_level1_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "cate_level2_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "cate_level3_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "cate_level4_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "country": v1.FixedLenFeature(1, tf.string, '-1')

           # , "seq_cate_id": v1.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
           # , "seq_goods_id": v1.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
           # , "seq_cate_id": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
           # , "seq_goods_id": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
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

    score[ID] = []
    score[CTR] = []
    score[CVR] = []
    score[CTCVR] = []
    score[CLK] = []
    score[PAY] = []
    for file_n, file in enumerate(tfr_list):
        # print('download file into tmp:',file)
        os.system('aws s3 cp %s %s' % (file, tmp_dir_data))
        file_suffix = tmp_dir_data + file.split('/')[-1]
        ds = tf.data.TFRecordDataset(file_suffix)
        ds = ds.map(_parse_fea).batch(batch_size)
        item_features_string = {"goods_id": "", "cate_id": "", "cate_level1_id": "", "cate_level2_id": "",
                                "cate_level3_id": "", "cate_level4_id": "", "country": ""
            ,"prop_seaon":"-1","prop_length":"-1","prop_main_material":"-1", "prop_pattern":"-1","prop_style":"-1"
                                ,"prop_quantity":"-1","prop_fitness":"-1",
                                "main_goods_id":"-1", "main_cate_id":"-1","main_cate_level2_id":"-1","main_cate_level3_id":"-1"
                                ,"main_cate_level4_id":"-1"
                                }
        item_features_double = {"ctr_7d": 0.0, "cvr_7d": 0.0}
        item_features_int = {"show_7d": 0, "click_7d": 0, "cart_7d": 0, "ord_total": 0, "pay_total": 0, "ord_7d": 0,
                             "pay_7d": 0,
                             "is_rel_cate":0, "is_rel_cate2":0, "is_rel_cate3":0,"is_rel_cate4":0, "sales_price":0
                             }
        user_seq_string = {"highLevelSeqListGoods": [""] * 20,
                           "highLevelSeqListCateId": [""] * 20, "lowerLevelSeqListGoods": [""] * 20,
                           "lowerLevelSeqListCateId": [""] * 20,
                           "last_login_device":"-1",
                           "last_login_brand":"-1",
                           "register_brand":"-1",
                           }
        predictor = tf.saved_model.load(dir).signatures["serving_default"]
        for idx in ds.as_numpy_iterator():
            feed_dict = {}
            id = idx[ID].tolist()
            id = [e[0] for e in id]
            score[ID].extend(id)

            # is_clk
            clk = idx[CLK].tolist()
            clk = [e[0] for e in clk]
            score[CLK].extend(clk)
            # is_pay
            pay = idx[PAY].tolist()
            pay = [e[0] for e in pay]
            score[PAY].extend(pay)

            for name in item_features_string.keys():
                feed_dict[name] = tf.constant(idx[name], dtype=tf.string)
            for name in item_features_int.keys():
                feed_dict[name] = tf.constant(idx[name], dtype=tf.int64)
            for name in item_features_double.keys():
                feed_dict[name] = tf.constant(idx[name], dtype=tf.float32)
            for name in user_seq_string.keys():
                feed_dict[name] = tf.constant(idx[name], dtype=tf.string)
            if debug:
                print('feed_dict:', feed_dict)
            res = predictor(**feed_dict)
            if debug:
                print('red:', res)

            prob = res[CTR].numpy().tolist()
            prob = [e[0] for e in prob]
            score[CTR].extend(prob)
            # print('res', res)
            cvr = res[CVR].numpy().tolist()
            cvr = [e[0] for e in cvr]
            score[CVR].extend(cvr)
            # ctcvr
            ctcvr = res[CTCVR].numpy().tolist()
            ctcvr = [e[0] for e in ctcvr]
            score[CTCVR].extend(ctcvr)
        # print('rm file:',file_suffix)
        print('proc %s process file:%s / %s' % (str(proc), str(file_n), str(len(tfr_list))))
        os.system('rm %s'%file_suffix)

def main(args):
    s3_cli = boto3.client('s3')
    BUCKET = 'warehouse-algo'

    # get files
    paginator = s3_cli.get_paginator('list_objects_v2')
    page_iter = paginator.paginate(Bucket=BUCKET, Prefix=args.tfr_s3)
    file_list = [[v['Key'] for v in page.get('Contents', [])] for page in page_iter][0]
    if args.sample_num is not None:
        print('sample file num:', args.sample_num)
        file_list = ['s3://%s/%s'%(BUCKET, v) for v in file_list][0:args.sample_num]
    else:
        file_list = ['s3://%s/%s' % (BUCKET, v) for v in file_list]
    print('file list num in dir', len(file_list))
    shuffle(file_list)
    file_batch = list(chunks(file_list,  args.proc))

    # download model
    s3_model = prod_model + args.model_name + args.model_version
    model_local = tmp_dir + args.model_name + args.model_version
    os.system("rm -rf %s" % (model_local))
    os.system("mkdir -p %s" % (model_local))
    os.system("aws s3 cp --recursive %s %s" % (s3_model, model_local))

    st = time.time()
    manager = multiprocessing.Manager()
    score = manager.dict()
    # score[ID] = []
    # score[SCORE] = []
    jobs = []
    # for thread_idx, tfr_list in enumerate(file_batch):
    #     score[thread_idx] = manager.dict()
    #     score[thread_idx][ID] = manager.list()
    #     score[thread_idx][CTR] = manager.list()
    #     score[thread_idx][CVR] = manager.list()
    #     score[thread_idx][CTCVR] = manager.list()
    #     score[thread_idx][CLK] = manager.list()
    #     score[thread_idx][PAY] = manager.list()
    #     p = multiprocessing.Process(target=process_tfr, args=(thread_idx, tfr_list, args.batch_size, model_local, score))
    #     jobs.append(p)
    #     p.start()
    # for proc in jobs:
    #     proc.join()
    d1 = {}
    p1 = multiprocessing.Process(target=process_tfr, args=(0, file_batch[0], args.batch_size, model_local, d1))
    jobs.append(p1)
    p1.start()
    d2 = {}
    p2 = multiprocessing.Process(target=process_tfr, args=(1, file_batch[1], args.batch_size, model_local, d2))
    jobs.append(p2)
    p2.start()
    d3 = {}
    p3 = multiprocessing.Process(target=process_tfr, args=(2, file_batch[2], args.batch_size, model_local, d3))
    jobs.append(p3)
    p3.start()
    d4 = {}
    p4 = multiprocessing.Process(target=process_tfr, args=(3, file_batch[3], args.batch_size, model_local, d4))
    jobs.append(p4)
    p4.start()
    d5 = {}
    p5 = multiprocessing.Process(target=process_tfr, args=(4, file_batch[4], args.batch_size, model_local, d5))
    jobs.append(p5)
    p5.start()
    d6 = {}
    p6 = multiprocessing.Process(target=process_tfr, args=(5, file_batch[5], args.batch_size, model_local, d6))
    jobs.append(p6)
    p6.start()
    d7 = {}
    p7 = multiprocessing.Process(target=process_tfr, args=(6, file_batch[6], args.batch_size, model_local, d7))
    jobs.append(p7)
    p7.start()
    d8 = {}
    p8 = multiprocessing.Process(target=process_tfr, args=(7, file_batch[7], args.batch_size, model_local, d8))
    jobs.append(p8)
    p8.start()
    d9 = {}
    p9 = multiprocessing.Process(target=process_tfr, args=(8, file_batch[8], args.batch_size, model_local, d9))
    jobs.append(p9)
    p9.start()
    d10 = {}
    p10 = multiprocessing.Process(target=process_tfr, args=(9, file_batch[9], args.batch_size, model_local, d10))
    jobs.append(p10)
    p10.start()
    for proc in jobs:
        proc.join()


    ed = time.time()
    print('multi predict done cost:', str(ed - st))

    # merge multi thread score
    merge_score = dict()
    st = time.time()
    print('d1', d1)
    for k, v in d1.items():
        print(k, v[0:10])
        if k in merge_score:
            merge_score[k].extend(v)
        else:
            merge_score[k] = v

    for k, v in d2.items():
        if k in merge_score:
            merge_score[k].extend(v)
        else:
            merge_score[k] = v

    for k, v in d3.items():
        if k in merge_score:
            merge_score[k].extend(v)
        else:
            merge_score[k] = v
    for k, v in d4.items():
        if k in merge_score:
            merge_score[k].extend(v)
        else:
            merge_score[k] = v
    for k, v in d5.items():
        if k in merge_score:
            merge_score[k].extend(v)
        else:
            merge_score[k] = v
    for k, v in d6.items():
        if k in merge_score:
            merge_score[k].extend(v)
        else:
            merge_score[k] = v
    for k, v in d7.items():
        if k in merge_score:
            merge_score[k].extend(v)
        else:
            merge_score[k] = v
    for k, v in d8.items():
        if k in merge_score:
            merge_score[k].extend(v)
        else:
            merge_score[k] = v
    for k, v in d9.items():
        if k in merge_score:
            merge_score[k].extend(v)
        else:
            merge_score[k] = v
    for k, v in d10.items():
        if k in merge_score:
            merge_score[k].extend(v)
        else:
            merge_score[k] = v
    ed = time.time()
    print('end merge score cost: %s'%(str(ed - st)))

    # save
    st = time.time()
    tb = pa.table(merge_score)
    save_file = pred_dir + args.model_name
    parquet.write_table(tb, save_file)
    ed = time.time()
    print('end write2table cost:', str(ed - st))

    st = time.time()
    pcvr, is_pay = [], []
    for e1, e2, e3 in zip(merge_score[CLK], merge_score[PAY], merge_score[CVR]):
        if e1 == 1:
            is_pay.append(e2)
            pcvr.append(e3)
    avg_pred_cvr = np.mean(pcvr)
    avg_label_pay = np.mean(is_pay)
    print('*' * 40)
    print('model_name:', args.model_name)
    print('model_version:', args.model_version)
    print('tfr_s3:', args.tfr_s3)
    print('N:', len(pcvr), 'avg_pred_cvr:', avg_pred_cvr, 'avg_label_pay:', avg_label_pay)
    auc = roc_auc_score(is_pay, pcvr)
    print('cvr-auc:', auc)
    ed = time.time()
    print('compute cvr-auc cost:', str(ed - st))

    # auc
    st = time.time()
    pctr = merge_score[CTR]
    is_clk = merge_score[CLK]
    avg_pred_ctr = np.mean(pctr)
    avg_label_clk = np.mean(is_clk)
    print('N:',len(pctr), 'avg_pred_ctr:', avg_pred_ctr, 'avg_label_clk:', avg_label_clk)
    auc = roc_auc_score(list(is_clk), list(pctr))
    print('ctr-auc:', auc)
    ed = time.time()
    print('compute ctr-auc cost:', str(ed - st))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='predict',
        description='predict',
        epilog='predict')
    parser.add_argument('--model_name', default='prod_mtl_seq_on_esmm_v1')
    parser.add_argument('--model_version', default='/ds=20241202-20241209/model/1734033700')
    parser.add_argument('--tfr', default='./part-00000-1186234f-fa44-44a8-9aff-08bcf2c5fb26-c000')
    parser.add_argument('--tfr_s3', default='rec/cn_rec_detail_sample_v10_tfr/ds=20241210/')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--proc', type=int, default=10)
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    debug = args.debug
    main(args)
