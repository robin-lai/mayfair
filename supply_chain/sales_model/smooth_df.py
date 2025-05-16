import pandas as pd
from pyarrow import parquet

in_file = 's3://warehouse-algo/sc_forecast_sequence_ts_model_train_and_predict_skc/ds=20250515/'
pt = parquet.read_table(in_file).to_pandas()
print(pt.head(10))
