# encoding:utf-8
import tensorflow as tf
import tensorflow.compat.v1 as v1

feature_spec_serve = {
    # double 类型字段
    "pctr_14d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pctr_14d"),
    "pcart_14d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pcart_14d"),
    "pwish_14d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pwish_14d"),
    "pcvr_14d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pcvr_14d"),
    "pctr_30d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pctr_30d"),
    "pcart_30d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pcart_30d"),
    "pwish_30d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pwish_30d"),
    "pcvr_30d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pcvr_30d"),

    "mt_i2i_main": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="mt_i2i_main"),
    "mt_i2i_long": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="mt_i2i_long"),
    "mt_i2i_short": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="mt_i2i_short"),
    "mt_hot": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="mt_hot"),
    "mt_hot_i2leaf": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="mt_hot_i2leaf"),
    "mt_i2i_main_score": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="mt_i2i_main_score"),
    "mt_i2i_long_score": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="mt_i2i_long_score"),
    "mt_i2i_short_score": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="mt_i2i_short_score"),

    "sales_price": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="sales_price"),

    "main_goods_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="main_goods_id"),
    "main_cate_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="main_cate_id"),
    "main_cate_level2_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="main_cate_level2_id"),
    "main_cate_level3_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="main_cate_level3_id"),
    "main_cate_level4_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="main_cate_level4_id"),

    "goods_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="goods_id"),
    "cate_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_id"),
    "cate_level2_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_level2_id"),
    "cate_level3_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_level3_id"),
    "cate_level4_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_level4_id"),
}

feature_describe = {
    "pctr_14d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_14d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_14d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_14d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pctr_30d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_30d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_30d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_30d": tf.FixedLenFeature(1, tf.float32, -1.0),

    "mt_i2i_main": tf.FixedLenFeature(1, tf.int64, 0),
    "mt_i2i_long": tf.FixedLenFeature(1, tf.int64, 0),
    "mt_i2i_short": tf.FixedLenFeature(1, tf.int64, 0),
    "mt_hot_i2leaf": tf.FixedLenFeature(1, tf.int64, 0),
    "mt_hot": tf.FixedLenFeature(1, tf.int64, 0),
    "mt_i2i_main_score": tf.FixedLenFeature(1, tf.float32, -1.0),
    "mt_i2i_long_score": tf.FixedLenFeature(1, tf.float32, -1.0),
    "mt_i2i_short_score": tf.FixedLenFeature(1, tf.float32, -1.0),

    "sales_price": tf.FixedLenFeature(1, tf.int64, 0),
    "main_goods_id": tf.FixedLenFeature(1, tf.string, "-1"),
    "main_cate_id": tf.FixedLenFeature(1, tf.string, "-1"),
    "main_cate_level2_id": tf.FixedLenFeature(1, tf.string, "-1"),
    "main_cate_level3_id": tf.FixedLenFeature(1, tf.string, "-1"),
    "main_cate_level4_id": tf.FixedLenFeature(1, tf.string, "-1"),

    "cate_id": tf.FixedLenFeature(1, tf.string, "-1"),
    "goods_id": tf.FixedLenFeature(1, tf.string, "-1"),
    "cate_level2_id": tf.FixedLenFeature(1, tf.string, "-1"),
    "cate_level3_id": tf.FixedLenFeature(1, tf.string, "-1"),
    "cate_level4_id": tf.FixedLenFeature(1, tf.string, "-1"),

    "is_clk": tf.FixedLenFeature(1, tf.int64, 0),
    "is_pay": tf.FixedLenFeature(1, tf.int64, 0),
    "sample_id": tf.FixedLenFeature(1, tf.string, "-1"),
}
