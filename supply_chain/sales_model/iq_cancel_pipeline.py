import datetime
import os

import pandas as pd

from common import *
import argparse
from pipeline import *

time_delta = 0
train_and_predict_data_path = "sc_forecast_sequence_ts_model_train_and_predict_skc_iq/"
base_dir = "./data_cancel_iq/"
suffix = 'iq'
model_num = 1

train_and_predict_data_path_smooth = base_dir + "sc_forecast_sequence_ts_model_train_and_predict_skc_%s_smooth.csv" % suffix
train_and_predict_data_path_smooth_eval = base_dir + "sc_forecast_sequence_ts_model_train_and_predict_skc_%s_smooth_eval.csv" % suffix
saved_model_path = base_dir + "best_model.pth"
s3_saved_model_path = 's3://warehouse-algo/sequence_model_predict_best_model_%s/ds=%s/'
s3_pred_result = 's3://warehouse-algo/sequence_model_predict_result_%s/ds=%s/'
s3_evaluated_result_path = 's3://warehouse-algo/sequence_model_evaluated_result_%s/ds=%s/evaluated_result.parquet'
os.system('rm -rf %s' % base_dir)
os.system('mkdir %s' % base_dir)

local_train_data_path = base_dir + "sequence_data.csv"
local_metrics_path = base_dir + "metrics_skc_diff.csv"
tmp_path = base_dir + 'tmp.txt'
local_future_dau_plan_path = base_dir + "savana_future_daus.csv"
local_evaluated_result_path = base_dir + "evaluated_result.parquet"
local_predicted_result_path = base_dir + 'output.parquet'
local_predict_dir = base_dir + 'pred/'
os.system('mkdir %s' % local_predict_dir)





def main(args):
    st1 = time.time()
    print('step1:process data')
    dc = DataConfig("sales_1d", time_delta)

    # step 1: process data
    if 'init' in args.pipeline:
        init(dc,train_and_predict_data_path,local_train_data_path, tmp_path)

    if 'train' in args.pipeline:
        train(dc)

    if 'pred' in args.pipeline:
        pred(dc)

    if 'eval' in args.pipeline:
        eval(dc)

    if 'metrics' in args.pipeline:
        metrics()

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
