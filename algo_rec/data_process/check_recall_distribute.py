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
        s3_file = s3_buk + buket + file
        os.system('aws s3 cp %s %s' % (s3_file, local_file))
        ll.append(local_file)
    return ll

def recall_ana(d):
    n =  len(d['sample_id'])
    print('n:',n)
    recall = {}
    for ss in d['s']:
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
    for file in file_ll:
        with open(file, 'rb') as fin:
            d.update(pickle.load(fin))
    recall_ana(d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='',
        description='',
        epilog='')
    parser.add_argument('--stat_file', default='rec/cn_rec_detail_sample_v30_savana_in_tfr_stat/ds=20250114/')
    args = parser.parse_args()
    main(args)
