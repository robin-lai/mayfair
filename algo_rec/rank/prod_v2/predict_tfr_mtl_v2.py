import argparse
from heapq import merge
from datetime import datetime,timedelta
import pprint
import pyarrow as pa
import copy
import traceback
from random import shuffle
import json

from pyarrow import parquet
import tensorflow as tf
import sys

from rope.base.builtins import get_file

from feature_serv_describe_tfv2 import feature_describe_pred

print(tf.__version__)
if '2' not in tf.__version__:
    print('use tf2')
    sys.exit(0)
import tensorflow.compat.v1 as v1
import multiprocessing
import boto3
import os
import time
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
import tensorflow.compat.v1 as v1

import math


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    c = math.ceil(len(lst) / n)
    for i in range(0, len(lst), c):
        yield lst[i:i + c]


ID = 'sample_id'
mt = 'mt'
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


def process_tfr(proc, tfr_list, batch_size, dir, pkl_file, site_code):
    def _parse_fea(data):
        feature_describe_pred.update(
            {"highLevelSeqListGoods": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20),
             "highLevelSeqListCateId": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20),
             "lowerLevelSeqListGoods": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20),
             "lowerLevelSeqListCateId": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20),
             "highLevelSeqList_len": v1.FixedLenFeature(1, tf.int64, default_value=0),
             "lowerLevelSeqList_len": v1.FixedLenFeature(1, tf.int64, default_value=0),
             }
        )

        features = tf.io.parse_single_example(data, features=feature_describe_pred)
        return features

    os.system('mkdir -p %s' % tmp_dir_data)
    score = {}
    score[ID] = []
    score[CTR] = []
    score[CVR] = []
    # score[CTCVR] = []
    score[CLK] = []
    score[PAY] = []
    for file_n, file in enumerate(tfr_list):
        # print('download file into tmp:',file)
        os.system('aws s3 cp %s %s' % (file, tmp_dir_data))
        file_suffix = tmp_dir_data + file.split('/')[-1]
        ds = tf.data.TFRecordDataset(file_suffix)
        ds = ds.map(_parse_fea)
        if site_code is not None:
            print('only site_code:%s data use' % (str(site_code)))
            ds = ds.filter(lambda x: tf.math.equal(x['country'][0], site_code))
        ds = ds.batch(batch_size)

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

            for name, v in feature_describe_pred.items():
                if name in ['is_clk', 'is_pay', 'sample_id']:
                    continue
                if 'tf.string' in str(v):
                    feed_dict[name] = tf.constant(idx[name], dtype=tf.string)
                if 'tf.int64' in str(v):
                    feed_dict[name] = tf.constant(idx[name], dtype=tf.int64)
                if 'tf.float32' in str(v):
                    feed_dict[name] = tf.constant(idx[name], dtype=tf.float32)
            print(f"feed_dict fts num:{len(feed_dict)}")
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
            # if CTCVR in res:
            #     ctcvr = res[CTCVR].numpy().tolist()
            #     ctcvr = [e[0] for e in ctcvr]
            #     score[CTCVR].extend(ctcvr)
        # print('rm file:',file_suffix)
        print('proc %s process file:%s / %s' % (str(proc), str(file_n), str(len(tfr_list))))
        os.system('rm %s' % file_suffix)
    with open(pkl_file, 'wb') as fout:
        pickle.dump(score, fout)


def auc_fun(label, pre):
    new_data = [[p, l] for p, l in zip(pre, label)]
    new_data.sort(key=lambda x: x[0])
    score_index = {}
    for index, i in enumerate(new_data):
        if i[0] not in score_index:
            score_index[i[0]] = []
        score_index[i[0]].append(index + 1)
    rank_sum = 0.
    for i in new_data:
        if i[1] == 1:
            rank_sum += sum(score_index[i[0]]) / len(score_index[i[0]]) * 1.0
    pos = label.count(1)
    neg = label.count(0)
    if not pos or not neg:
        return None
    return (rank_sum - (pos * (pos + 1) * 0.5)) / (pos * neg)


def gauc_fun(pred_d, label_idx, pre_idx, type):
    gauc = {}
    gauc_l = []
    none_auc = 0
    try:
        for u, l in pred_d.items():
            pred = [e[pre_idx] for e in l]
            label = [e[label_idx] for e in l]
            auc_score = auc_fun(label, pred)
            if auc_score is not None:
                gauc[u] = auc_score
                gauc_l.append(auc_score)
            else:
                none_auc += 1
                # print('uid:%s auc is none'%(u), l)
    except Exception:
        print('data:', l)
        traceback.print_exc(file=sys.stdout)
    gnum = len(pred_d.keys())
    gpos = len(gauc_l)
    gneg = none_auc
    gauc = np.mean(gauc_l)
    pp = [10, 20, 30.40, 50, 60, 70, 80, 90, 100]
    gaucpp = np.percentile(gauc_l, pp)
    print('none_auc num %s of all %s :%s' % (str(gneg), type, str(gnum)))
    print('%s num:%s have auc' % (type, str(gpos)))
    print('type:%s' % type, gauc)
    print('type:%s percentle:' % type, gaucpp)
    return gnum, gpos, gneg, gauc, gaucpp


def get_file_batch(args):
    s3_cli = boto3.client('s3')
    BUCKET = 'warehouse-algo'

    # get files
    paginator = s3_cli.get_paginator('list_objects_v2')
    page_iter = paginator.paginate(Bucket=BUCKET, Prefix=args.tfr_s3)
    file_list = [[v['Key'] for v in page.get('Contents', [])] for page in page_iter][0]
    if args.sample_num is not None:
        print('sample file num:', args.sample_num)
        file_list = ['s3://%s/%s' % (BUCKET, v) for v in file_list][0:args.sample_num]
    else:
        file_list = ['s3://%s/%s' % (BUCKET, v) for v in file_list]
    print('file list num in dir', len(file_list))
    shuffle(file_list)
    file_batch = list(chunks(file_list, args.proc))
    return file_batch


def multi_process(args, file_batch):
    # download model
    s3_model = prod_model + args.model_name + args.model_version
    model_local = tmp_dir + args.model_name + args.model_version
    predict_local = tmp_dir + args.model_name
    os.system("rm -rf %s" % (model_local))
    os.system("mkdir -p %s" % (model_local))
    os.system("mkdir -p %s" % (predict_local))
    os.system("aws s3 cp --recursive %s %s" % (s3_model, model_local))

    st = time.time()
    jobs = []
    predict_files = []
    for thread_idx, tfr_list in enumerate(file_batch):
        pkl_file = '%s/predict_part_%s.pkl' % (str(predict_local), str(thread_idx))
        predict_files.append(pkl_file)
        p = multiprocessing.Process(target=process_tfr,
                                    args=(thread_idx, tfr_list, args.batch_size, model_local, pkl_file, args.site_code))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    ed = time.time()
    print('multi predict done cost:', str(ed - st))
    return predict_files


def merge_pred_score(args, predict_files):
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
    ed = time.time()
    print('end merge score cost: %s' % (str(ed - st)))
    dump_file = './' + args.model_name + '_' + args.model_version.split('/')[-1] + '.pkl'
    with open(dump_file, 'wb') as fout:
        pickle.dump(merge_score, fout)
    print('write pred score2file:', dump_file)
    return merge_score


def compute_metrics(merge_score):

    model_info = {}
    model_info['model_name'] = args.model_name
    model_info['version'] = args.model_version
    tfr_s3_tt = args.tfr_s3.split('/')
    model_info['ds'] = tfr_s3_tt[-2]
    model_info['sample'] = tfr_s3_tt[-3]
    now = datetime.now()
    model_info['datetime'] = now.strftime("%Y-%m-%d-%H:%M:%S")

    # auc
    auc_ctr_d = copy.deepcopy(model_info)
    st = time.time()
    pctr = merge_score[CTR]
    is_clk = merge_score[CLK]
    avg_pred_ctr = np.mean(pctr)
    avg_label_clk = np.mean(is_clk)
    print('N:', len(pctr), 'avg_pred_ctr:', avg_pred_ctr, 'avg_label_clk:', avg_label_clk)
    auc = roc_auc_score(list(is_clk), list(pctr))
    print('ctr-auc:', auc)
    ed = time.time()
    auc_ctr_d['n'] = str(len(pctr))
    auc_ctr_d['n+'] = str(np.sum(is_clk))
    auc_ctr_d['n-'] = str(len(pctr) - np.sum(is_clk))
    auc_ctr_d['pred'] = str(avg_pred_ctr)
    auc_ctr_d['label'] = str(avg_label_clk)
    auc_ctr_d['auc'] = str(auc)
    auc_ctr_d['type'] = 'all-ctr'
    print('compute ctr-auc cost:', str(ed - st))

    # prepare data
    pred = []
    uuid_pred = {}
    req_pred = {}
    for id, clk, pay, ctr, cvr in zip(merge_score['sample_id'], merge_score['is_clk'], merge_score['is_pay'],
                                      merge_score['ctr'], merge_score['cvr']):
        # "concat(bhv.country,'|',bhv.scene_code,'|',bhv.client_type,'|',bhv.uuid,'|',bhv.pssid,'|',bhv.recid,'|',bhv.main_goods_id,'|',bhv.goods_id)ASsample_id,"
        token = str(id).split('|')
        uuid, reqid = token[3], token[5]

        pred.append((uuid, reqid, clk, pay, ctr, cvr))
        tt = (clk, pay, ctr, cvr)
        if uuid in uuid_pred:
            uuid_pred[uuid].append(tt)
        else:
            uuid_pred[uuid] = [tt]
        if reqid in req_pred:
            req_pred[reqid].append(tt)
        else:
            req_pred[reqid] = [tt]
    # cvr
    label = [e[2] for e in pred]
    pre = [e[4] for e in pred]
    auc_all = auc_fun(label, pre)
    print('N:', len(pred), 'label_mean:', np.mean(label), 'pred_mean:', np.mean(pre), 'auc-all-ctr:', auc_all)
    label_cvr = [e[3] for e in pred if e[2] == 1]
    pre_cvr = [e[5] for e in pred if e[2] == 1]
    auc_all_cvr = auc_fun(label_cvr, pre_cvr)
    print('N:', len(label_cvr), 'label_mean:', np.mean(label_cvr), 'pred_mean:', np.mean(pre_cvr), 'auc-all-ctr:',
          auc_all_cvr)

    auc_cvr_d = copy.deepcopy(model_info)
    auc_cvr_d['n'] = str(len(label_cvr))
    auc_cvr_d['n+'] = str(np.sum(label_cvr))
    auc_cvr_d['n-'] = str(len(label_cvr) - np.sum(label_cvr))
    auc_cvr_d['pred'] = str(np.mean(pre_cvr))
    auc_cvr_d['label'] = str(np.mean(label_cvr))
    auc_cvr_d['auc'] = str(auc_all_cvr)
    auc_cvr_d['type'] = 'all-cvr'

    # uauc_ctr
    gauc_ctr_user_d = copy.deepcopy(model_info)
    print('uuid num:', len(uuid_pred.keys()))
    ugnum, ugpos, ugneg, ugauc, ugaucpp = gauc_fun(uuid_pred, 0, 3, 'u-ctr-gauc')
    gauc_ctr_user_d['n'] = str(ugnum)
    gauc_ctr_user_d['n+'] = str(ugpos)
    gauc_ctr_user_d['n-'] = str(ugneg)
    gauc_ctr_user_d['auc'] = str(ugauc)
    gauc_ctr_user_d['auc-pp'] = ','.join([str(e) for e in ugaucpp])
    gauc_ctr_user_d['type'] = 'uuid_gauc'

    # qauc_ctr
    gauc_ctr_req_d = copy.deepcopy(model_info)
    print('recid num:', len(req_pred.keys()))
    qgnum, qgpos, qgneg, qgauc, qgaucpp = gauc_fun(req_pred, 0, 3, 'q-ctr-gauc')
    gauc_ctr_req_d['n'] = str(qgnum)
    gauc_ctr_req_d['n+'] = str(qgpos)
    gauc_ctr_req_d['n-'] = str(qgneg)
    gauc_ctr_req_d['auc'] = str(qgauc)
    gauc_ctr_req_d['auc-pp'] = ','.join([str(e) for e in qgaucpp])
    gauc_ctr_req_d['type'] = 'recid_gauc'

    # save auc
    auc_local_file = './auc.json'
    os.system('rm %s' % auc_local_file)
    os.system("aws s3 cp %s %s" % (args.auc_file, auc_local_file))
    auc_list = [auc_ctr_d, auc_cvr_d, gauc_ctr_user_d, gauc_ctr_req_d]
    print('*' * 60)
    with open('./auc.json', 'r') as fin:
        js = json.load(fin)
    auc_list.extend(list(js))
    with open('./auc.json', 'w') as fout:
        json.dump(auc_list, fout, sort_keys=True)
    if debug:
        print('debug model skip cp auc file to aws')
    else:
        os.system("aws s3 cp %s %s" % (auc_local_file, args.auc_file))

    pprint.pprint(json.dumps(auc_list))
    # with open(auc_local_file, 'wb') as fout:
    #     pickle.dump(auc_list, fout)
    # with open(auc_local_file, 'rb') as fin:
    #     auc_list = pickle.load(fin)


def get_model_version(prefix):
    print(f"prefix:{prefix}")
    import boto3
    s3_cli = boto3.client('s3')
    BUCKET = 'warehouse-algo'
    paginator = s3_cli.get_paginator('list_objects_v2')
    page_iter = paginator.paginate(Bucket=BUCKET,
                                   Prefix=prefix)
    file_list = [[v['Key'] for v in page.get('Contents', [])] for page in page_iter]
    model_version = [e.split('/')[-2] for e in file_list[0] if 'saved_model' in e]
    print(f"modelversion:{model_version[0]}")
    return model_version[0]


def main(args):
    file_batch = get_file_batch(args)
    predict_files = multi_process(args, file_batch)
    merge_score = merge_pred_score(args, predict_files)
    compute_metrics(merge_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='predict',
        description='predict',
        epilog='predict-use tf2.0')
    parser.add_argument('--model_name', default='mtl_seq_esmm')
    parser.add_argument('--ds', type=str,
                        default=(datetime.today() - timedelta(days=2)).strftime('%Y%m%d'))
    parser.add_argument('--model_version', default='/ds=%s/model/%s/')
    parser.add_argument('--tfr', default='')
    parser.add_argument('--tfr_s3', default='rec/cn_rec_detail_sample_v30_savana_in_tfr/ds=%s/')
    parser.add_argument('--auc_file', default='s3://warehouse-algo/rec/model_pred/auc.json')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--proc', type=int, default=10)
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--site_code', type=str, default=None)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    args.tfr_s3 = args.tfr_s3 % args.ds
    print(f"tfr_s3:{args.tfr_s3}")
    version = get_model_version('rec/prod_model/%s/ds=%s/model/' % (args.model_name, args.ds))
    args.model_version = args.model_version % (args.ds, version)
    debug = args.debug
    st = time.time()
    main(args)
    ed = time.time()
    print('cost', str(ed - st))
