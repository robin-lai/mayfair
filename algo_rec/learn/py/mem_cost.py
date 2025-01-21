from pympler.asizeof import asizeof

from pyarrow import parquet
import os
s3_file = 's3://warehouse-algo/rec/cn_rec_detail_sample_v20_savana_in/ds=20250120/part-00000-7c77b5e6-3306-42b4-a878-1a0712ce3ab4-c000'
local_file = './tmp/part-00000-7c77b5e6-3306-42b4-a878-1a0712ce3ab4-c000'
os.system('aws s3 cp %s %s' % (s3_file, local_file))

pt = parquet.read_table(local_file)
pt2 = parquet.read_table(local_file).to_pylist()

print(f"pt mem size mem [M]", asizeof(pt) / 1048576)
print(f"pt2 mem size mem [M]", asizeof(pt2) / 1048576)

