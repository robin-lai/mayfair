import argparse
import os
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
import traceback
import datetime

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
                vs = []
                for tt in v.split(chr(2)):
                    vs.append(tt.split(chr(4)))
                i2i_d[k] = vs
    print('read i2i end, num:', len(i2i_d.keys()))
    return i2i_d

def run_multi_process(func,args):
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
    proc_list = [multiprocessing.Process(target=func, args=(args[0], args[1], args[2],args.i2i_part, args.i2i_s3, args.i2i_file)) for args in args_list]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]
    fail_cnt = sum([p.exitcode for p in proc_list])
    if fail_cnt > 0:
        print('process fail cnt:', fail_cnt)
        # raise ValueError('Failed in %d process.' % fail_cnt)

def gen_fts(path_pt_list, path_tfr_local_list, path_tfr_s3_list,i2i_part, i2i_s3, i2i_file ):
    i2i_d = get_i2i(i2i_part, i2i_s3, i2i_file)
    for pt_file, tfr_local_file, tfr_s3_file in zip(path_pt_list, path_tfr_local_list, path_tfr_s3_list):
        pt = parquet.read_table(pt_file).to_pylist()
        for tt in pt:
            mt = []
            main_goods = tt['main_goods_id']
            tgt_id = tt['goods_id']
            seq_on = tt['seq_on']
            if main_goods in i2i_d:
                if tgt_id in i2i_d[main_goods]:
                    mt.append('i2i_main')
            if seq_on == "":
                js = dict()
            else:
                js = dict(json.loads(seq_on))
            low_ids = seq_on[]
            high_ids = seq_on['highLevelSeqList']


def main(args):
    run_multi_process(gen_fts, args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gentfr',
        description='gentfr',
        epilog='gentfr-help')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--ds', type=str, default='ds=%s'%(datetime.date.today() - datetime.timedelta(days=2)).strftime('%Y%m%d'))
    parser.add_argument('--range', type=str, default='')
    parser.add_argument('--i2i_s3',
                        default='s3://warehouse-algo/rec/recall/cn_rec_detail_recall_i2i_for_redis%s/item_user_debias_%s/')
    parser.add_argument('--i2i_file', default='swing_rec_Savana_IN_part_%s')
    parser.add_argument('--i2i_part', type=int, default=7)
    parser.add_argument('--thread', type=int, default=15)
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--dir_pt', default='cn_rec_detail_sample_v20_savana_in/')
    parser.add_argument('--dir_tfr', default='cn_rec_detail_sample_v20_savana_in_tfr/')

