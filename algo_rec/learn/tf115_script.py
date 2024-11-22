import tensorflow as tf

seq_cate_id_bytes = [bytes(v, encoding="utf8") for v in ["5", "6", "7", "8", "9", "10"]]
print('seq_cate_id_bytes', seq_cate_id_bytes)
my_example = tf.train.Example(features=tf.train.Features(feature={
    'my_ints': tf.train.Feature(int64_list=tf.train.Int64List(value=[5, 6, 7, 8, 9])),
    'my_ints_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=[5, 6, 7, 8, 9,10])),
    'seq_cate_id': tf.train.Feature(bytes_list=tf.train.BytesList(
        value=seq_cate_id_bytes))
    # 'seq_cate_id': tf.train.Feature(bytes_list=tf.train.BytesList(
    #     value=["5", "6", "7", "8", "9", "10"]))
}))
print('my_example', my_example)

my_example_str = my_example.SerializeToString()
with tf.python_io.TFRecordWriter('my_example.tfrecords') as writer:
    writer.write(my_example_str)

# Reading it back via a Dataset

featuresDict = {
    'my_ints': tf.io.FixedLenFeature(5, dtype=tf.int64),
    'my_ints_seq': tf.io.FixedLenFeature(6, dtype=tf.int64),
    "seq_cate_id": tf.io.FixedLenFeature(6, dtype=tf.string, default_value=["", "", "", "", "", ""])
                }


def parse_tfrecord(example):
    features = tf.parse_single_example(example, featuresDict)
    return features


Dataset = tf.data.TFRecordDataset('my_example.tfrecords')

Dataset = Dataset.map(parse_tfrecord).batch(1)

print('features:', Dataset)
print(['*'] * 40)
tf.print(Dataset)
print(['*'] * 40)
iterator = Dataset.make_one_shot_iterator()
with tf.Session() as sess:
    print(sess.run(iterator.get_next()))
