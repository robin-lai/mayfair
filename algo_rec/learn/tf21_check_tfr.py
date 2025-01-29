import tensorflow as tf

tfrecord_file = "part-00000-4e93eb14-9da8-475f-b889-7ff7ca761cfe-c000"

raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

for raw_record in raw_dataset.take(5):  # 读取前5条记录
    try:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())  # 解析 TFRecord
        print(example)
    except Exception as e:
        print(f"Error parsing record: {e}")
