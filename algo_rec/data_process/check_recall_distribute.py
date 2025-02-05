import argparse
import os
import pickle

import boto3

def get_bucket_files(local_dir, buket):
    s3_cli = boto3.client('s3')
    BUCKET = 'warehouse-algo'
    s3_obj = "rec/"
    s3_buk = "s3://warehouse-algo/"
    paginator = s3_cli.get_paginator('list_objects_v2')
    page_iter = paginator.paginate(Bucket=BUCKET, Prefix=buket)
    file_list = [[v['Key'] for v in page.get('Contents', [])] for page in page_iter][0]
    ll = []
    for file in file_list:
        local_file = local_dir + file
        s3_file = s3_buk + file
        os.system('aws s3 cp %s %s' % (s3_file, local_file))
        ll.append(local_file)
    print(f"file list {ll}")
    return ll

def recall_ana(d):
    recall = {}
    clk_recall = {}
    n = 0
    clk_n = 0
    not_recall_n = 0
    for i in d.keys():
        s_len = len(d[i]['s'])
        n += s_len
        print(f"s_len:{s_len}")
        if len(d[i]['s']) < 1:
            not_recall_n += 1

        for ss, is_clk in zip(d[i]['s'], d[i]['is_clk']):
            for s in ss:
                if int(is_clk) == 1:
                    clk_n += 1
                    if s in clk_recall:
                        clk_recall[s] += 1
                    else:
                        clk_recall[s] = 0
                if s in recall:
                    recall[s] += 1
                else:
                    recall[s] = 0
    print(f"not_recall_n:{not_recall_n} ratio:{not_recall_n / n}")
    for k, v in recall.items():
        print(f"曝光口径 s:{k}, v:{v} n:{n} ratio:{v / n}")
    for k, v in clk_recall.items():
        print(f"点击口径 s:{k}, v:{v} n:{clk_n} ratio:{v / clk_n}")


def main(args):
    file_ll = get_bucket_files('./tmp/', args.stat_file)
    d = {}
    for i, file in enumerate(file_ll[1:]): # first is base dir
        with open(file, 'rb') as fin:
            d[i] = pickle.load(fin)
    recall_ana(d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='',
        description='',
        epilog='')
    parser.add_argument('--stat_file', default='rec/cn_rec_detail_sample_v30_savana_in_tfr_stat/ds=%s/')
    parser.add_argument('--ds', default='20250126')
    args = parser.parse_args()
    args.stat_file = args.stat_file % args.ds
    print(f"stat_file:{args.stat_file}")
    main(args)
