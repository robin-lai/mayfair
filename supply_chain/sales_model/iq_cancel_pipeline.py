import datetime
import os
from common import *
import argparse

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
tmp_path = base_dir + 'tmp.txt'
local_future_dau_plan_path = base_dir + "savana_future_daus.csv"
local_evaluated_result_path = base_dir + "evaluated_result.parquet"
local_predicted_result_path = base_dir + 'output.parquet'
local_predict_dir = base_dir + 'pred/'
os.system('mkdir %s' % local_predict_dir)

def main(args):
    st1 = time.time()
    st = time.time()
    print('step1:process data')
    dc = DataConfig("sales_1d", time_delta)
    yesterday_str = dc.yesterday.strftime("%Y%m%d")

    if 'init' in args.pipeline:
        # step 1: process data
        wait_for_ready(train_and_predict_data_path, dc.yesterday.strftime("%Y%m%d"))
        ret = list(load_s3_dir(BUCKET, train_and_predict_data_path, [dc.yesterday.strftime("%Y%m%d")], tmp_path))
        ret = pd.DataFrame(ret)
        ret = ret[ret["target_date"] >= (dc.today - timedelta(days=250)).strftime("%Y-%m-%d")]
        print(ret["target_date"].max(), ret["target_date"].min())
        ret.to_csv(local_train_data_path)
        del ret
        ed = time.time()
        print('process data cost:', ed - st)

    if 'train' in args.pipeline:
        ed = time.time()
        print('step2: train')
        dc.init_df(local_train_data_path)
        train_loader, test_loader = prepare_train_valid_data(dc, (dc.today - timedelta(days=0)).strftime('%Y-%m-%d'),
                                                             train_and_predict_data_path_smooth)
        train(dc, train_loader, test_loader, saved_model_path)
        os.system('aws s3 cp %s %s' % (saved_model_path, s3_saved_model_path % (suffix, yesterday_str)))
        print('train cost:', time.time() - ed)

    if 'pred' in args.pipeline:
        print('step3: pred')
        ed = time.time()
        for i in range(0, model_num):
            download_date = dc.yesterday - timedelta(days=i)
            download_date = download_date.strftime("%Y%m%d")
            remote_path = s3_saved_model_path % (suffix, download_date) + "best_model_iq.pth"
            print('remote_path', remote_path)
            download_file(remote_path, local_predict_dir + "best_model_%s.pth" % download_date)
        #
        predicted_result = daily_predict(dc, local_predict_dir)
        predicted_result.to_parquet(local_predicted_result_path)
        os.system('aws s3 cp %s %s' % (local_predicted_result_path, s3_pred_result % (suffix, yesterday_str)))
        print('pred cost:', time.time() - ed)

    if 'eval' in args.pipeline:
        ed = time.time()
        print('step4: evalute')
        evaluate_model(dc, local_evaluated_result_path, local_predict_dir, train_and_predict_data_path_smooth_eval)
        os.system('aws s3 cp %s %s' % (local_evaluated_result_path, s3_evaluated_result_path % (suffix, yesterday_str)))
        print('evalute cost:', time.time() - ed)

    if 'metrics' in args.pipeline:
        print('step5: 评测')
        pred_date_str = '20250511'
        pred_date = datetime.strptime(pred_date_str, "%Y%m%d").date()
        pred_df = parquet.read_table(s3_pred_result % (suffix, pred_date_str)).to_pandas().drop_duplicates()
        print(pred_df.describe())
        real_df_file = 's3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc_smooth_iq/ds=20250514/'
        real_df =  parquet.read_table(real_df_file).to_pandas()[['target_date', 'skc_id', 'sales_1d', 'no_cancel_sales_1d']].astype({"target_date":str, "skc_id": int, "sales_1d": int, "no_cancel_sales_1d":int})
        real_df['target_date'] = pd.to_datetime(real_df['target_date'])
        print(real_df.describe())
        print(pred_df.head())
        print(real_df.head())
        pred_ids = set(pred_df['skc_id'].values)
        real_ids = set(real_df['skc_id'].values)
        ids = pred_ids.intersection(real_ids)
        for id in ids:
            print('id', id)
            id_df = pred_df[pred_df['skc_id'] == id]
            w1_p = id_df[id_df['week_num'] == 1]['predict'].values[0]
            w2_p = id_df[id_df['week_num'] == 2]['predict'].values[0]
            w3_p = id_df[id_df['week_num'] == 3]['predict'].values[0]
            w4_p = id_df[id_df['week_num'] == 4]['predict'].values[0]
            r_df = real_df[real_df['skc_id'] == id]
            end_w1 = pred_date + pd.Timedelta(days=7)
            end_w2 = pred_date + pd.Timedelta(days=14)
            end_w3 = pred_date + pd.Timedelta(days=21)
            end_w4 = pred_date + pd.Timedelta(days=28)
            w1_r = r_df[r_df['target_date']].between(pred_date, end_w1)
            print(w1_r)
            w2_r = r_df[r_df['target_date']].between(pred_date, end_w2)
            print(w2_r)
            w3_r = r_df[r_df['target_date']].between(pred_date, end_w3)
            print(w3_r)
            w4_r = r_df[r_df['target_date']].between(pred_date, end_w4)
            print(w4_r)





    print('all cost[hour]:', (time.time() - st1) / 3600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='sc_iq',
        description='sc_iq',
        epilog='sc-iq-help')
    parser.add_argument('--pipeline', type=str,
                        default='init,train,pred,eval,metrics')
    args = parser.parse_args()
    # main(args)
    print('step5: 评测')
    pred_df = parquet.read_table(s3_pred_result % (suffix, '20250511')).to_pandas()
    print(pred_df.describe())
    real_df_file = 's3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc_smooth_iq/ds=20250514/'
    real_df = parquet.read_table(real_df_file).to_pandas()
    print(real_df.describe())
    print(pred_df.head())
    print(real_df.head())


