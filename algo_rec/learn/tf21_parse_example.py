import tensorflow as tf
print(tf.__version__)
file = "/home/sagemaker-user/mayfair/algo_rec/data/cn_rec_detail_sample_v1_ctr/ds=20241102/part-00000-84854856-90d1-4bf8-9e52-5032827fdee4-c000"
ds = tf.data.TFRecordDataset(file)
import tensorflow.compat.v1 as v1

def parse(data):
    feature_describe = {
           # "ctr_7d": v1.FixedLenFeature(1, v1.float32, 0.0)
           # , "cvr_7d": v1.FixedLenFeature(1, v1.float32, 0.0)
           # , "show_7d": v1.FixedLenFeature(1, v1.int64, 0)
           # , "click_7d": v1.FixedLenFeature(1, v1.int64, 0)
           # , "cart_7d": v1.FixedLenFeature(1, v1.int64, 0)
           # , "ord_total": v1.FixedLenFeature(1, v1.int64, 0)
           # , "pay_total": v1.FixedLenFeature(1, v1.int64, 0)
           # , "ord_7d": v1.FixedLenFeature(1, v1.int64, 0)
           # , "pay_7d": v1.FixedLenFeature(1, v1.int64, 0)

           # , "cate_id": v1.FixedLenFeature(1, v1.string, "-1")
           # , "goods_id": v1.FixedLenFeature(1, v1.string, "-1")
           # , "cate_level1_id": v1.FixedLenFeature(1, v1.string, "-1")
           # , "cate_level2_id": v1.FixedLenFeature(1, v1.string, "-1")
           # , "cate_level3_id": v1.FixedLenFeature(1, v1.string, "-1")
           # , "cate_level4_id": v1.FixedLenFeature(1, v1.string, "-1")
           # , "country": v1.FixedLenFeature(1, v1.string, '-1')

           # # , "seq_cate_id": v1.FixedLenSequenceFeature(20, v1.string, default_value="-1", allow_missing=True)
           # # , "seq_goods_id": v1.FixedLenSequenceFeature(20, v1.string, default_value="-1", allow_missing=True)
           # , "seq_cate_id": v1.FixedLenFeature(20, v1.string, default_value=[""] * 20)
           # , "seq_goods_id": v1.FixedLenFeature(20, v1.string, default_value=[""] * 20)

            "is_clk": v1.FixedLenFeature(1, v1.int64, 0)
           , "is_pay": v1.FixedLenFeature(1, v1.int64, 0)
           # , "sample_id": v1.FixedLenFeature(1, v1.string, "-1")
       }
    features = tf.io.parse_single_example(data, features=feature_describe)
    return features

ds = ds.map(parse)
print(list(ds.take(1000).as_numpy_iterator()))



