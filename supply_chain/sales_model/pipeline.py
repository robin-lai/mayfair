import datetime
import os

import pandas as pd

from common import *
import argparse
def init(dc, data_path, local_data_path, tmp_path):
    st = time.time()
    wait_for_ready(data_path, dc.yesterday.strftime("%Y%m%d"))
    ret = list(load_s3_dir(BUCKET, data_path, [dc.yesterday.strftime("%Y%m%d")], tmp_path))
    ret = pd.DataFrame(ret)
    print(ret.head())
    ret = ret[ret["target_date"] >= (dc.today - timedelta(days=250)).strftime("%Y-%m-%d")]
    print(ret["target_date"].max(), ret["target_date"].min())
    ret.to_csv(local_data_path)
    del ret
    ed = time.time()
    print('process data cost:', ed - st)


def train_pipeline(dc, model_path, s3_model_path):
    ed = time.time()
    print('step2: train')
    train_loader, test_loader = prepare_train_valid_data(dc, (dc.today - timedelta(days=0)).strftime('%Y-%m-%d'))
    train(dc, train_loader, test_loader, model_path)
    os.system('aws s3 cp %s %s' % (model_path, s3_model_path % (dc.yesterday.strftime("%Y%m%d"))))
    print('train cost:', time.time() - ed)


def pred(dc, model_num, s3_model_path, local_pred_dir, local_pred_path, s3_pred_path):
    print('step3: pred')
    ed = time.time()
    for i in range(0, model_num):
        download_date = dc.yesterday - timedelta(days=i)
        download_date = download_date.strftime("%Y%m%d")
        remote_path = s3_model_path % download_date + "best_model.pth"
        print('remote_path', remote_path)
        download_file(remote_path, local_pred_dir + "best_model_%s.pth" % download_date)
    #
    predicted_result = daily_predict(dc, local_pred_dir)
    predicted_result.to_parquet(local_pred_path)
    os.system(
        'aws s3 cp %s %s' % (local_pred_path, s3_pred_path % (dc.yesterday.strftime("%Y%m%d"))))
    print('pred cost:', time.time() - ed)


def eval(dc, local_eval_path, local_pred_dir, data_smooth_eval_path, s3_eval_path):
    ed = time.time()
    print('step4: evalute')
    evaluate_model(dc, local_eval_path, local_pred_dir, data_smooth_eval_path)
    os.system('aws s3 cp %s %s' % (
        local_eval_path, s3_eval_path % dc.yesterday.strftime("%Y%m%d")))
    print('evalute cost:', time.time() - ed)


def metrics(s3_pred_path, local_metrics_path):
    print('step5: 评测')
    pred_date_str = '2025-05-11'
    pred_date = datetime.strptime(pred_date_str, "%Y-%m-%d")
    pred_df = parquet.read_table(s3_pred_path % pred_date_str.replace('-','')).to_pandas().drop_duplicates()
    print(pred_df.describe())
    real_df_file = 's3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc_smooth_iq/ds=%s/' % pred_date_str.replace(
        '-', '')
    real_df = parquet.read_table(real_df_file).to_pandas()[
        ['target_date', 'skc_id', 'sales_1d', 'no_cancel_sales_1d']].astype(
        {"target_date": str, "skc_id": int, "sales_1d": int, "no_cancel_sales_1d": int})
    real_df['target_date'] = pd.to_datetime(real_df['target_date'])
    print(real_df.describe())
    print(pred_df.head())
    print(real_df.head())
    pred_ids = set(pred_df['skc_id'].values)
    real_ids = set(real_df['skc_id'].values)
    ids = pred_ids.intersection(real_ids)
    ret_list = []
    for id in ids:
        print('id', id)
        id_df = pred_df[pred_df['skc_id'] == id]
        w1_p = id_df[id_df['week_num'] == 1]['predict'].values[0]
        w2_p = id_df[id_df['week_num'] == 2]['predict'].values[0]
        w3_p = id_df[id_df['week_num'] == 3]['predict'].values[0]
        w4_p = id_df[id_df['week_num'] == 4]['predict'].values[0]
        r_df = real_df[real_df['skc_id'] == id].dropna()
        end_w1 = pred_date + pd.Timedelta(days=7)
        end_w2 = pred_date + pd.Timedelta(days=14)
        end_w3 = pred_date + pd.Timedelta(days=21)
        end_w4 = pred_date + pd.Timedelta(days=28)
        w1_r = r_df[r_df['target_date'].between(pred_date, end_w1)]['sales_1d'].values
        w2_r = r_df[r_df['target_date'].between(pred_date, end_w2)]['sales_1d'].values
        w3_r = r_df[r_df['target_date'].between(pred_date, end_w3)]['sales_1d'].values
        w4_r = r_df[r_df['target_date'].between(pred_date, end_w4)]['sales_1d'].values
        w1_diff, w2_diff, w3_diff, w4_diff = -1, -1, -1, -1
        w1_r2, w2_r2, w3_r2, r4_r2 = -1, -1, -1, -1
        w1_l, w2_l, w3_l, w4_l = len(w1_r), len(w2_r), len(w3_r), len(w4_r)
        if len(w1_r) > 0 and sum(w1_r) > 0:
            w1_r2 = np.mean(w1_r) * 7
            w1_diff = np.abs(w1_p - w1_r2) / w1_r2
        if len(w2_r) > 0 and sum(w2_r) > 0:
            w2_r2 = np.mean(w2_r) * 7
            w2_diff = np.abs(w2_p - w2_r2) / w2_r2
        if len(w3_r) > 0 and sum(w3_r) > 0:
            w3_r2 = np.mean(w3_r) * 7
            w3_diff = np.abs(w3_p - w3_r2) / w3_r2
        if len(w4_r) > 0 and sum(w4_r) > 0:
            w4_r2 = np.mean(w4_r) * 7
            w4_diff = np.abs(w4_p - w4_r2) / w4_r2
        ret_list.append(
            [id, w1_p, w2_p, w3_p, w4_p, w1_r2, w2_r2, w3_r2, r4_r2, w1_diff, w2_diff, w3_diff, w4_diff, w1_l, w2_l,
             w3_l, w4_l])
        print(
            f"id:{id}, w1_p:{w1_p}, w2_p:{w2_p}, w3_p:{w3_p}, w4_p:{w4_p},  w1_r2:{w1_r2}, w2_r2:{w2_r2}, w3_r2:{w3_r2}, r4_r2:{r4_r2}, "
            f" w1_diff:{w1_diff}, w2_diff:{w2_diff}, w3_diff:{w3_diff}, w4_diff:{w4_diff},  w1_l:{w1_l}, w2_l:{w2_l}, w3_l:{w3_l}, w4_l:{w4_l} ")
    ret_df = pd.DataFrame(ret_list,
                          columns=['id', 'w1_p', 'w2_p', 'w3_p', 'w4_p', 'w1_r2', 'w2_r2', 'w3_r2', 'r4_r2',
                                   'w1_diff', 'w2_diff', 'w3_diff', 'w4_diff', 'w1_l', 'w2_l', 'w3_l', 'w4_l'])
    ret_df.to_csv(local_metrics_path)