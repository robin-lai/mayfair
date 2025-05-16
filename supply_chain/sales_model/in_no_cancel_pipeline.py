import datetime
import os

import pandas as pd

from common import *
import argparse
from pipeline import *

time_delta = 1
base_dir = "./data_no_cancel/"
data_path = "sc_forecast_sequence_ts_model_train_and_predict_skc/"
data_smooth_eval_path = base_dir + "sc_forecast_sequence_ts_model_train_and_predict_skc_smooth_eval.csv"
model_num = 1

model_path = base_dir + "best_model.pth"
s3_model_path = 's3://warehouse-algo/sequence_model_predict_best_model/ds=%s/'
s3_pred_path = 's3://warehouse-algo/sequence_model_predict_result/ds=%s/'
s3_eval_path = 's3://warehouse-algo/sequence_model_evaluated_result/ds=%s/evaluated_result.parquet'
os.system('rm -rf %s' % base_dir)
os.system('mkdir %s' % base_dir)

local_data_path = base_dir + "sequence_data.csv"
local_metrics_path = base_dir + "metrics_skc_diff.csv"
tmp_path = base_dir + 'tmp.txt'
local_future_dau_plan_path = base_dir + "savana_future_daus.csv"
local_eval_path = base_dir + "evaluated_result.parquet"
local_pred_path = base_dir + 'output.parquet'
local_pred_dir = base_dir + 'pred/'
os.system('mkdir %s' % local_pred_dir)


def main(args):
    st1 = time.time()
    print('step1:process data')
    dc = DataConfig("no_cancel_sales_1d", time_delta)

    # step 1: process data
    if 'init' in args.pipeline:
        init(dc, data_path, local_data_path, tmp_path)

    if 'train' in args.pipeline:
        train_pipeline(dc, model_path, s3_model_path)

    if 'pred' in args.pipeline:
        pred(dc, model_num, s3_model_path, local_pred_dir, local_pred_path, s3_pred_path)

    if 'eval' in args.pipeline:
        eval(dc, local_eval_path, local_pred_dir, data_smooth_eval_path, s3_eval_path)

    if 'metrics' in args.pipeline:
        metrics(s3_pred_path, local_metrics_path)

    print('all cost[hour]:', (time.time() - st1) / 3600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='sc_iq',
        description='sc_iq',
        epilog='sc-iq-help')
    parser.add_argument('--pipeline', type=str,
                        default='init,train,pred,eval,metrics')
    args = parser.parse_args()
    main(args)
