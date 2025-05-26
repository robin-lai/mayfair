from pyarrow import parquet
import os
import pandas as pd
import time
from common import smooth_flash
import argparse


def main(args):
    st = time.time()
    df = parquet.read_table(args.in_file).to_pandas()
    codes = set(df['skc_id'].values.tolist())
    count = 0
    df_list = []
    for code in list(codes):
        if count % 500 == 0:
            print("step: %d / %d" % (count, len(codes)))
        count += 1
        sub_df = df[df['skc_id'] == code].copy(deep=True)
        sub_df = sub_df.sort_values(by='target_date')
        sub_df = smooth_flash(sub_df)
        df_list.append(sub_df)

    df2 = pd.concat(df_list)
    local_file_csv = './tmp/' + '_'.join(args.s3_file.split('/'))
    local_file_pt = './tmp/' + '_'.join(args.s3_file.split('/')) + '.parquet'
    df2.to_csv(local_file_csv)
    df2.to_parquet(local_file_pt, engine="pyarrow")
    os.system('aws s3 cp %s %s' % (local_file_pt, args.s3_file))
    print(df2.head(10))
    print('cost:', time.time() - st)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='fun',
        description='fun',
        epilog='fun-help')
    parser.add_argument('--ds', type=str,default='')
    parser.add_argument('--range', type=str,default='20250501,20250502,20250503,20250504,20250505,20250506,20250507,20250508,20250509,20250510,20250512,20250513,20250515,20250516,20250518,20250519,20250520,20250521,20250522,20250523,20250524,20250525')
    parser.add_argument('--in_file', type=str,default='s3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc_iq/ds=%s/')
    parser.add_argument('--s3_file', type=str,default='s3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc_smooth_iq/ds=%s/')
    args = parser.parse_args()
    # in
    # site_in_file = 's3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc/ds=20250515/'
    # site_s3_file = 's3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc_smooth/ds=20250515/'
    # smooth_df(in_file=site_in_file, s3_file=site_s3_file)

    # iq
    if args.range != '':
        for ds in args.range.split(','):
            args.in_file = args.in_file % ds
            args.s3_file = args.s3_file % ds
            main(args)
    else:
        args.in_file = args.in_file % args.ds
        args.s3_file = args.s3_file % args.ds
        main(args)

