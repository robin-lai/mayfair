from pyarrow import parquet
import os
import pandas as pd
import time
from common import smooth_flash


def smooth_df(in_file, s3_file):
    st = time.time()
    df = parquet.read_table(in_file).to_pandas()
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
    local_file_csv = './tmp/' + '_'.join(s3_file.split('/'))
    local_file_pt = './tmp/' + '_'.join(s3_file.split('/')) + '.parquet'
    df2.to_csv(local_file_csv)
    df2.to_parquet(local_file_pt, engine="pyarrow")
    os.system('aws s3 cp %s %s' % (local_file_pt, s3_file))
    print(df2.head(10))
    print('cost:', time.time() - st)


if __name__ == '__main__':
    # in
    # site_in_file = 's3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc/ds=20250515/'
    # site_s3_file = 's3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc_smooth/ds=20250515/'
    # smooth_df(in_file=site_in_file, s3_file=site_s3_file)

    # iq
    site_in_file = 's3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc_iq/ds=20250514/'
    site_s3_file = 's3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc_smooth_iq/ds=20250514/'
    smooth_df(in_file=site_in_file, s3_file=site_s3_file)
