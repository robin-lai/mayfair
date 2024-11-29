import tensorflow as tf
print(tf.__version__)
file = "/home/sagemaker-user/mayfair/algo_rec/data/cn_rec_detail_sample_v1_cvr/ds=20241102/part-00000-84854856-90d1-4bf8-9e52-5032827fdee4-c000"
ds = tf.data.TFRecordDataset(file)


def parse(data):
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
           , "sample_id": tf.FixedLenFeature(1, tf.string, "-1")
       }
    features = tf.io.parse_single_example(data, features=feature_describe)
    return features

ds = ds.map(parse)
print(list(ds.take(10).as_numpy_iterator()))



