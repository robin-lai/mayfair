import tensorflow as tf
import tensorflow.compat.v1 as v1

tfrecord_file = "part-00000-4e93eb14-9da8-475f-b889-7ff7ca761cfe-c000"

raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

for raw_record in raw_dataset.take(5):  # 读取前5条记录
    try:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())  # 解析 TFRecord
        print(example)
    except Exception as e:
        print(f"Error parsing record: {e}")


example = tf.train.Example(features=tf.train.Features(feature={
    "mt": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"some_data"])),
}))

with tf.io.TFRecordWriter("output.tfrecord") as writer:
    writer.write(example.SerializeToString())

    def parse(data):
        feature_describe = {
            "mt": v1.FixedLenFeature([], tf.string, default_value=""),

        }
        features = tf.io.parse_single_example(data, features=feature_describe)
        return features

    ds = tf.data.TFRecordDataset("output.tfrecord")
    ds = ds.map(parse).batch(10)
    print(list(ds.as_numpy_iterator()))
