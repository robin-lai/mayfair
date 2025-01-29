import tensorflow as tf
import tensorflow.compat.v1 as v1
from babel.messages.frontend import extract_messages

tfrecord_file = "part-00000-4e93eb14-9da8-475f-b889-7ff7ca761cfe-c000"

raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

for raw_record in raw_dataset.take(5):  # 读取前5条记录
    try:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())  # 解析 TFRecord
        print(example)
    except Exception as e:
        print(f"Error parsing record: {e}")


# example = tf.train.Example(features=tf.train.Features(feature={
#     "mt": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"some_data"])),
# }))
feature = {}
def bytes_fea(v_list, n=1, encode=False):
    v_list = v_list if isinstance(v_list, list) else [v_list]
    if len(v_list) > n:
        v_list = v_list[:n]
    elif len(v_list) < n:
        v_list.extend([""] * (n - len(v_list)))
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[bytes(v, encoding="utf8") for v in v_list]))
feature["mt"] = bytes_fea(["i2i", "u2i"], n=3)
sample = tf.train.Example(features=tf.train.Features(feature=feature))
record = sample.SerializeToString()

with tf.io.TFRecordWriter("output.tfrecord") as writer:
    writer.write(record)

def parse(data):
    feature_describe = {
        "mt": v1.FixedLenFeature(3, tf.string, default_value=[""] * 3),

    }
    features = tf.io.parse_single_example(data, features=feature_describe)
    return features

ds = tf.data.TFRecordDataset("output.tfrecord")
ds = ds.map(parse).batch(10)
print('1', list(ds.as_numpy_iterator()))
try:
    example = tf.train.Example()
    example.ParseFromString(ds.numpy())  # 解析 TFRecord
    print('2', example)
except Exception as e:
    print(f"Error parsing record: {e}")
