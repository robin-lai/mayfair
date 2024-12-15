import argparse
from heapq import merge

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
import pickle
from sklearn.metrics import roc_auc_score

import math
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    c = math.ceil(len(lst) / n)
    for i in range(0, len(lst), c):
        yield lst[i:i + c]


ID = 'uuid'
PROB = 'probabilities'
CLK = 'is_clk'
prod_model = 's3://warehouse-algo/rec/'
pred_dir = 's3://warehouse-algo/rec/qrqm_model_test/'
tmp_dir = '/home/sagemaker-user/tmp/'
tmp_dir_data = tmp_dir + 'data/'

debug = False

def process_tfr(proc, tfr_list, batch_size, dir, pkl_file):

    def _parse_fea(data):
       feature_describe = {
           "uuid": v1.FixedLenFeature(1, tf.string, "-1")
           , "age": v1.FixedLenFeature(1, tf.string, "-1")
           , "site_code": v1.FixedLenFeature(1, tf.string, "-1")
           , "model_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "height": v1.FixedLenFeature(1, tf.string, "-1")
           , "weight": v1.FixedLenFeature(1, tf.string, "-1")
           , "bust": v1.FixedLenFeature(1, tf.string, "-1")
           , "waistline": v1.FixedLenFeature(1, tf.string, "-1")
           , "hips": v1.FixedLenFeature(1, tf.string, "-1")
           , "shoulder_width": v1.FixedLenFeature(1, tf.string, "-1")
           , "arm_length": v1.FixedLenFeature(1, tf.string, "-1")
           , "thigh_circumference": v1.FixedLenFeature(1, tf.string, "-1")
           , "calf_circumference": v1.FixedLenFeature(1, tf.string, "-1")
           , "pgid": v1.FixedLenFeature(1, tf.string, "-1")
           , "goods_id": v1.FixedLenFeature(1, tf.string, "-1")
           , "is_clk": v1.FixedLenFeature(1, tf.int64, 0)
       }
       features = tf.io.parse_single_example(data, features=feature_describe)
       return features
    os.system('mkdir -p %s'%tmp_dir_data)
    score = {}
    score[ID] = []
    score[CLK] = []
    score[PROB] = []
    for file_n, file in enumerate(tfr_list):
        # print('download file into tmp:',file)
        os.system('aws s3 cp %s %s' % (file, tmp_dir_data))
        file_suffix = tmp_dir_data + file.split('/')[-1]
        ds = tf.data.TFRecordDataset(file_suffix)
        ds = ds.map(_parse_fea).batch(batch_size)
        item_features_string = {"uuid": "", "age": "", "site_code": "", "model_id": "","height":"",
                                "weight":"", "bust":"", "waistline":"", "hips":"", "shoulder_width":"",
                                "arm_length":"", "thigh_circumference":"", "calf_circumference":"", "pgid":"",
                                "goods_id":""
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

            for name in item_features_string.keys():
                feed_dict[name] = tf.constant(idx[name], dtype=tf.string)
            if debug:
                print('feed_dict:', feed_dict)
            res = predictor(**feed_dict)
            prob = res[PROB].numpy().tolist()
            prob = [e[0] for e in prob]
            score[PROB].extend(prob)
            if debug:
                print('red:', res)

            # print('res', res)
        # print('rm file:',file_suffix)
        print('proc %s process file:%s / %s' % (str(proc), str(file_n), str(len(tfr_list))))
        os.system('rm %s'%file_suffix)
    with open(pkl_file, 'wb') as fout:
        pickle.dump(score, fout)

def main(args):
    s3_cli = boto3.client('s3')
    BUCKET = 'algo-sg'

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
    jobs = []
    predict_files = []
    for thread_idx, tfr_list in enumerate(file_batch):
        pkl_file = './predict_part_%s.pkl'%(str(thread_idx))
        predict_files.append(pkl_file)
        p = multiprocessing.Process(target=process_tfr, args=(thread_idx, tfr_list, args.batch_size, model_local,pkl_file ))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()


    ed = time.time()
    print('multi predict done cost:', str(ed - st))

    # merge multi thread score
    merge_score = dict()
    st = time.time()
    for file in predict_files:
        with open(file, 'rb') as fin:
            d = pickle.load(fin)
            for k, v in d.items():
                if k in merge_score:
                    merge_score[k].extend(v)
                else:
                    merge_score[k] = v
    print('end merge score cost: %s'%(str(ed - st)))

    # save
    st = time.time()
    tb = pa.table(merge_score)
    save_file = pred_dir + args.model_name
    parquet.write_table(tb, save_file)
    ed = time.time()
    print('end write2table cost:', str(ed - st))

    # auc
    st = time.time()
    pctr = merge_score[PROB]
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
    parser.add_argument('--model_name', default='qrqm_model_test')
    parser.add_argument('--model_version', default='/model/1734183567')
    parser.add_argument('--tfr', default='./part-00000-1186234f-fa44-44a8-9aff-08bcf2c5fb26-c000')
    parser.add_argument('--tfr_s3', default='qrqm_algo/qrqm_uuid_pgid_bhv_sample_v3_tfr/ds=20241211')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--proc', type=int, default=5)
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    debug = args.debug
    main(args)
