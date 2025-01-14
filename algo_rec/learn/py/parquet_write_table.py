import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

# 创建示例数据
data = {
    "trig_goods_id": [25, 30, 35, 40, 45],
    "trig_goods_name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "num": [25, 30, 35, 40, 45],
    "cat2": ["US", "US", "UK", "UK", "CA"],
    "cat3": ["US", "US", "UK", "UK", "CA"],
    "leaf": ["US", "US", "UK", "UK", "CA"],
    "leaf_cn": ["US", "US", "UK", "UK", "CA"],
    "trig_pic_url": ["US", "US", "UK", "UK", "CA"],
    "tgt_goods_id": [25, 30, 35, 40, 45],
    "tgt_goods_name": ["US", "US", "UK", "UK", "CA"],
    "tgt_score":  [25.0, 30.0, 35.0, 40.0, 45.0],
    "tgt_pic_url": ["US", "US", "UK", "UK", "CA"],
    "tgt_num": [25, 30, 35, 40, 45],
    "tgt_cat2": ["US", "US", "UK", "UK", "CA"],
    "tgt_cat3": ["US", "US", "UK", "UK", "CA"],
    "tgt_leaf": ["US", "US", "UK", "UK", "CA"],
    "tgt_leaf_cn": ["US", "US", "UK", "UK", "CA"],
    "version": ["swing", "swing", "swing", "swing", "swing"],
}
df = pd.DataFrame(data)

# 将 Pandas DataFrame 转换为 PyArrow 表
table = pa.Table.from_pandas(df)

# 定义输出目录
output_dir = "s3://warehouse-algo/rec/recall/rec_detail_recall_swing_result_version/"

# 删除已存在的分区目录
if os.path.exists(output_dir):
    import shutil
    shutil.rmtree(output_dir)

# 按照 "country" 字段分区
pq.write_to_dataset(
    table,
    root_path=output_dir,
    partition_cols=["version"]
)

print(f"Partitioned table created at: {output_dir}")
