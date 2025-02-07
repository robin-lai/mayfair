# encoding:utf-8
import tensorflow as tf
import tensorflow.compat.v1 as v1

feature_describe_pred = {
    "pctr_14d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_14d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_14d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_14d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pctr_30d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_30d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_30d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_30d": v1.FixedLenFeature(1, tf.float32, -1.0),


    "sales_price": v1.FixedLenFeature(1, tf.int64, 0),

    "mt_i2i_main": v1.FixedLenFeature(1, tf.int64, 0),
    "mt_i2i_long": v1.FixedLenFeature(1, tf.int64, 0),
    "mt_i2i_short": v1.FixedLenFeature(1, tf.int64, 0),
    "mt_hot_i2leaf": v1.FixedLenFeature(1, tf.int64, 0),
    "mt_hot": v1.FixedLenFeature(1, tf.int64, 0),
    "mt_i2i_main_score": v1.FixedLenFeature(1, tf.float32, -1.0),
    "mt_i2i_long_score": v1.FixedLenFeature(1, tf.float32, -1.0),
    "mt_i2i_short_score": v1.FixedLenFeature(1, tf.float32, -1.0),
    "mt_u2i_f": v1.FixedLenFeature(1, tf.float32, 0),
    "mt": v1.FixedLenFeature(6, tf.string, default_value=[""] * 6),

    "main_goods_id": v1.FixedLenFeature(1, tf.string, "-1"),
    "main_cate_id": v1.FixedLenFeature(1, tf.string, "-1"),
    "main_cate_level2_id": v1.FixedLenFeature(1, tf.string, "-1"),
    "main_cate_level3_id": v1.FixedLenFeature(1, tf.string, "-1"),
    "main_cate_level4_id": v1.FixedLenFeature(1, tf.string, "-1"),

    "cate_id": v1.FixedLenFeature(1, tf.string, "-1"),
    "goods_id": v1.FixedLenFeature(1, tf.string, "-1"),
    "cate_level2_id": v1.FixedLenFeature(1, tf.string, "-1"),
    "cate_level3_id": v1.FixedLenFeature(1, tf.string, "-1"),
    "cate_level4_id": v1.FixedLenFeature(1, tf.string, "-1"),

    "is_clk": v1.FixedLenFeature(1, tf.int64, 0),
    "is_pay": v1.FixedLenFeature(1, tf.int64, 0),
    "sample_id": v1.FixedLenFeature(1, tf.string, "-1"),
}