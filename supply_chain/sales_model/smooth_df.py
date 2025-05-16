from pyarrow import parquet
import os
from common import smooth_flash


def smooth_df(in_file, s3_file):
    df = parquet.read_table(in_file).to_pandas()
    df2 = smooth_flash(df)
    local_file_csv = './tmp/' + '_'.join(s3_file.split('/'))
    local_file_pt = './tmp/' + '_'.join(s3_file.split('/') + '.parquet')
    df2.to_csv(local_file_csv)
    df2.to_parquet(local_file_pt, engine="pyarrow")
    os.system('aws cp %s %s' % (local_file_pt, s3_file))
    print(df2.head(10))


if __name__ == '__main__':
    site_in_file = 's3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc/ds=20250515/'
    site_s3_file = 's3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc_smooth/ds=20250515/'
    smooth_df(in_file=site_in_file, s3_file=site_s3_file)