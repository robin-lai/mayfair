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
    n = 0
    for i in d.keys():
        s_len = len(d[i]['s'])
        n += s_len
        print(f"s_len:{s_len}")
        for ss in d[i]['s']:
            for s in ss:
               if s in recall:
                   recall[s] += 1
               else:
                   recall[s] = 0
    for k, v in recall.items():
        print(f"s:{k}, ratio:{v / n}")


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
    parser.add_argument('--stat_file', default='rec/cn_rec_detail_sample_v30_savana_in_tfr_stat/ds=20240114/')
    args = parser.parse_args()
    main(args)
