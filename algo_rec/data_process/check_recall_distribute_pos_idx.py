import argparse
import os,sys
import pickle
import traceback
import pandas as pd

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

def recall_ana(d, ds):
    print(f"日期口径 {ds}")
    recall_stat = []
    recall = {}
    clk_recall = {}
    pos_idx_d = {"all_exp":[], "all_clk":[]}
    n = 0
    n_dup = 0
    clk_n = 0
    clk_n_dup = 0
    not_recall_n = 0
    not_recall_n_clk = 0
    for i in d.keys():
        s_len = len(d[i]['s'])
        n += s_len
        print(f"s_len:{s_len}")

        for ss, is_clk, pos_idx in zip(d[i]['s'], d[i]['is_clk'], d[i]['pos_idx']):
            pos_idx_d['all_exp'].append(pos_idx)
            if int(is_clk) == 1:
                pos_idx_d['all_clk'].append(pos_idx)
                clk_n += 1

            if "".join(ss) == '':
                not_recall_n += 1
                if int(is_clk) == 1:
                    not_recall_n_clk += 1
                continue
            for s in ss:
                if s == '':continue
                n_dup += 1
                skey = s + '_exp'
                if skey in pos_idx_d:
                    pos_idx_d[skey].append(pos_idx)
                else:
                    pos_idx_d[skey] = [pos_idx]
                if int(is_clk) == 1:
                    sclk_key = s + '_clk'
                    if sclk_key in pos_idx_d:
                        pos_idx_d[sclk_key].append(pos_idx)
                    else:
                        pos_idx_d[sclk_key] = [pos_idx]
                    clk_n_dup += 1
                    if s in clk_recall:
                        clk_recall[s] += 1
                    else:
                        clk_recall[s] = 1
                if s in recall:
                    recall[s] += 1
                else:
                    recall[s] = 1
    pp = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    import numpy as np

    recall_stat.append([ds, 'all', 'exp', not_recall_n, n, round(not_recall_n / n, 5),  np.percentile(pos_idx_d['all_exp'], pp)])
    print(f"曝光口径 not_recall_n:{not_recall_n} ratio:{not_recall_n / n}")
    print(f"点击口径 not_recall_n:{not_recall_n_clk} ratio:{not_recall_n_clk / clk_n}")
    recall_stat.append([ds, 'all', 'clk', not_recall_n_clk, clk_n, round(not_recall_n_clk / clk_n, 5),  np.percentile(pos_idx_d['all_clk'], pp)])
    for k, v in recall.items():
        print(f"曝光口径 s:{k}, v:{v} n:{n_dup} ratio:{v / n_dup}")
        recall_stat.append([ds, k, 'exp', v, n_dup, round(v / n_dup, 5),  np.percentile(pos_idx_d[k + '_exp'], pp)])
    for k, v in clk_recall.items():
        print(f"点击口径 s:{k}, v:{v} n:{clk_n_dup} ratio:{v / clk_n_dup}")
        recall_stat.append([ds, k, 'clk', v, clk_n_dup, round(v / clk_n_dup, 5),  np.percentile(pos_idx_d[k + '_clk'], pp)])
    print(f"recall_stat {recall_stat}")
    return recall_stat


def main(args):
    local_tmp_dir = './tmp/'
    os.system("rm -rf %s" % local_tmp_dir)
    file_ll = get_bucket_files(local_tmp_dir, args.stat_file)
    d = {}
    for i, file in enumerate(file_ll): # first is base dir
        if not str.endswith(file, '.pkl'):
            print(f"file {file} not end with .pkl")
            continue
        with open(file, 'rb') as fin:
            d[i] = pickle.load(fin)
    return recall_ana(d, args.ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='',
        description='',
        epilog='')
    parser.add_argument('--stat_file', default='rec/cn_rec_detail_sample_v30_savana_in_tfr_stat_row_n300/ds=%s/')
    parser.add_argument('--local_file', default='./recall_distribute.csv')
    parser.add_argument('--ds', default='20250126')
    parser.add_argument('--range', default='20250101,20250102,20250103,20250104,20250105,20250106,20250107,20250108,20250109,20250110,20250111,20250112,20250113,20250114,20250115,20250116,20250117,20250118,20250119')
    args = parser.parse_args()
    recall_stat = []
    if args.range != '':
        try:
            for ds in args.range.split(','):
                args = parser.parse_args()
                args.ds = ds
                args.stat_file = args.stat_file % args.ds
                print(f"stat_file:{args.stat_file}")
                recall_stat.extend(main(args))
        except Exception:
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)
    else:
        args.stat_file = args.stat_file % args.ds
        print(f"stat_file:{args.stat_file}")
        recall_stat = main(args)
    df = pd.DataFrame(recall_stat, columns=['ds','recall', 'koujing', 'fengzi', 'fengmu', 'ratio', 'pos_idx_pp'])
    df.to_csv(args.local_file)

