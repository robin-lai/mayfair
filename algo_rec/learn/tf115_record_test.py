import tensorflow as tf

# print(tf.__version__)
# a = tf.data.Dataset.range(1, 1000000)
# iter = a.map(lambda x: x + 1).shard(1,0).shuffle(10).batch(10).prefetch(10).make_one_shot_iterator()
# print(iter.get_next())


def _parse_fea(data):
   feature_describe = {
       "ctr_7d": tf.FixedLenFeature(1, tf.float32, 0.0)
       , "cvr_7d": tf.FixedLenFeature(1, tf.float32, 0.0)
       , "show_7d": tf.FixedLenFeature(1, tf.int64, 0)
       , "click_7d": tf.FixedLenFeature(1, tf.int64, 0)
       , "cart_7d": tf.FixedLenFeature(1, tf.int64, 0)
       , "ord_total": tf.FixedLenFeature(1, tf.int64, 0)
       , "pay_total": tf.FixedLenFeature(1, tf.int64, 0)
       , "ord_7d": tf.FixedLenFeature(1, tf.int64, 0)
       , "pay_7d": tf.FixedLenFeature(1, tf.int64, 0)

       , "cate_id": tf.FixedLenFeature(1, tf.string, "-1")
       , "goods_id": tf.FixedLenFeature(1, tf.string, "-1")
       , "cate_level1_id": tf.FixedLenFeature(1, tf.string, "-1")
       , "cate_level2_id": tf.FixedLenFeature(1, tf.string, "-1")
       , "cate_level3_id": tf.FixedLenFeature(1, tf.string, "-1")
       , "cate_level4_id": tf.FixedLenFeature(1, tf.string, "-1")
       , "country": tf.FixedLenFeature(1, tf.string, '-1')

       # , "seq_cate_id": tf.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
       # , "seq_goods_id": tf.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
       , "seq_cate_id": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)
       , "seq_goods_id": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)

       , "is_clk": tf.FixedLenFeature(1, tf.int64, 0)
       , "is_pay": tf.FixedLenFeature(1, tf.int64, 0)
   }
   features = tf.io.parse_single_example(data, features=feature_describe)

   is_clk = features.pop('is_clk')
   is_pay = features.pop('is_pay')
   input_feat_norm = features
   print('features_data', features)

   return features, is_clk


def input_fn_from_local_tfrecords(mode, channel=None, feature_description=None, label=None, batch_size=10, num_epochs=1,
             num_parallel_calls=8,
             shuffle_factor=10, prefetch_factor=20,
             fn_mode='', num_host=1, host_rank=0):
    print('Begin_input_fn channel', channel, '#' * 80)

    dataset = tf.data.TFRecordDataset('./tfrecord/part-00000-e0a162c9-7a56-40c0-ae3f-25f194a0751e-c000')
    # data_iter_before = dataset.make_one_shot_iterator()
    # print('raw sample:', data_iter_before.get_next())
    # https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#training-with-pipe-mode-using-pipemodedataset
    if fn_mode == 'MultiWorkerShard':
        dataset = dataset.shard(num_host, host_rank)
    dataset = dataset.map(_parse_fea, num_parallel_calls=num_parallel_calls)
    dataset = dataset.filter(lambda x, y: tf.math.equal(x['country'],'IN'))

    dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)
    dataset = dataset.batch(batch_size)

    if channel == 'eval':
        print('Begin read eval data sample 100000', '#' * 80)
        dataset = dataset.take(100000)

    if prefetch_factor > 0:
        dataset = dataset.prefetch(buffer_size=prefetch_factor)
    try:
        data_iter = dataset.make_one_shot_iterator()
    except AttributeError:
        data_iter = tf.compat.v1.data.make_one_shot_iterator(dataset)

    features, click = data_iter.get_next()
    print('raw features:', features)
    print('raw click:', click)
    return features, click

if __name__ == '__main__':
    features, click = input_fn_from_local_tfrecords(mode='train')
    with tf.Session() as sess:
        print(sess.run([features, click]))