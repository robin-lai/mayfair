import os
from common import *

time_delta = 1
train_and_predict_data_path = "sc_forecast_sequence_ts_model_train_and_predict_skc/"
base_dir = "./data_no_cancel/"
os.system('rm -rf %s' % base_dir)
os.system('mkdir %s' % base_dir)
train_and_predict_data_path_smooth = base_dir + "sc_forecast_sequence_ts_model_train_and_predict_skc_smooth.csv"
train_and_predict_data_path_smooth_eval = base_dir + "sc_forecast_sequence_ts_model_train_and_predict_skc_smooth_eval.csv"
s3_saved_model_path = 's3://warehouse-algo/sequence_model_no_cancel_predict_best_model/ds=%s/'
s3_pred_result = 's3://warehouse-algo/sequence_model_predict_no_cancel_result/ds=%s/'
s3_evaluated_result_path = 's3://warehouse-algo/sequence_model_evaluated_no_cancel_result/ds=%s/evaluated_result.parquet'

saved_model_path = base_dir + "best_model.pth"
local_train_data_path = base_dir + "sequence_data.csv"
tmp_path = base_dir + 'tmp.txt'
local_future_dau_plan_path = base_dir + "savana_future_daus.csv"
local_evaluated_result_path = base_dir + "evaluated_result.parquet"
local_predicted_result_path = base_dir + 'output.parquet'
local_predict_dir = base_dir + 'pred/'
os.system('mkdir %s' % local_predict_dir)
model_num = 2

if __name__ == '__main__':
    st1 = time.time()
    st = time.time()
    print('step1:process data')
    dc = DataConfig("no_cancel_sales_1d", time_delta)

    # step 1: process data
    yesterday_str = dc.yesterday.strftime("%Y%m%d")
    wait_for_ready(train_and_predict_data_path, dc.yesterday.strftime("%Y%m%d"))
    ret = list(load_s3_dir(BUCKET, train_and_predict_data_path, [dc.yesterday.strftime("%Y%m%d")], tmp_path))
    ret = pd.DataFrame(ret)
    ret = ret[ret["target_date"] >= (dc.today - timedelta(days=250)).strftime("%Y-%m-%d")]
    print(ret["target_date"].max(), ret["target_date"].min())
    ret.to_csv(local_train_data_path)
    del ret
    ed = time.time()
    print('process data cost:', ed - st)

    print('train')
    dc.init_df(local_train_data_path)
    train_loader, test_loader = prepare_train_valid_data(dc, (dc.today - timedelta(days=0)).strftime('%Y-%m-%d'))
    train(dc, train_loader, test_loader, saved_model_path)
    os.system('aws s3 cp %s %s' % (saved_model_path, s3_saved_model_path % yesterday_str))
    print('train cost:', time.time() - ed)

    print('pred')
    ed = time.time()
    for i in range(0, model_num):
        download_date = dc.yesterday - timedelta(days=i)
        download_date = download_date.strftime("%Y%m%d")
        remote_path = s3_saved_model_path % download_date + "best_model.pth"
        download_file(remote_path, local_predict_dir + "best_model_%s.pth" % download_date)

    predicted_result = daily_predict(dc, local_predict_dir)
    predicted_result.to_parquet(local_predicted_result_path)
    os.system('aws s3 cp %s %s' % (local_predicted_result_path, s3_pred_result % yesterday_str))
    print('pred cost:', time.time() - ed)
    ed = time.time()

    print('step4: evalute')
    evaluate_model(dc, local_evaluated_result_path, local_predict_dir, train_and_predict_data_path_smooth_eval)
    os.system('aws s3 cp %s %s' % (local_evaluated_result_path, s3_evaluated_result_path % yesterday_str))
    print('evalute cost:', time.time() - ed)
    print('all cost[hour]:', (time.time() - st1) / 3600)
