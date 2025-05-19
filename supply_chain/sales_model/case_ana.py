import pandas as pd
from pyarrow import parquet

df_v1 = parquet.read_table("s3://warehouse-algo/sequence_model_predict_result_d/ds=20250511_m1/ds=20250511/").to_pandas().drop_duplicates()
df_v2 = parquet.read_table("s3://warehouse-algo/sequence_model_predict_result_d/ds=20250511/ds=20250511/").to_pandas().drop_duplicates()

