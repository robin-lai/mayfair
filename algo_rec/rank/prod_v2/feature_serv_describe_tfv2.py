# encoding:utf-8
import tensorflow as tf
import tensorflow.compat.v1 as v1

feature_describe_pred = {
    "pv_1d": v1.FixedLenFeature(1, tf.int64, -1),
    "ipv_1d": v1.FixedLenFeature(1, tf.int64, -1),
    "cart_1d": v1.FixedLenFeature(1, tf.int64, -1),
    "wish_1d": v1.FixedLenFeature(1, tf.int64, -1),
    "pay_1d": v1.FixedLenFeature(1, tf.int64, -1),
    "pctr_1d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_1d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_1d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_1d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pv_3d": v1.FixedLenFeature(1, tf.int64, -1),
    "ipv_3d": v1.FixedLenFeature(1, tf.int64, -1),
    "cart_3d": v1.FixedLenFeature(1, tf.int64, -1),
    "wish_3d": v1.FixedLenFeature(1, tf.int64, -1),
    "pay_3d": v1.FixedLenFeature(1, tf.int64, -1),
    "pctr_3d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_3d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_3d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_3d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pv_5d": v1.FixedLenFeature(1, tf.int64, -1),
    "ipv_5d": v1.FixedLenFeature(1, tf.int64, -1),
    "cart_5d": v1.FixedLenFeature(1, tf.int64, -1),
    "wish_5d": v1.FixedLenFeature(1, tf.int64, -1),
    "pay_5d": v1.FixedLenFeature(1, tf.int64, -1),
    "pctr_5d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_5d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_5d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_5d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pv_7d": v1.FixedLenFeature(1, tf.int64, -1),
    "ipv_7d": v1.FixedLenFeature(1, tf.int64, -1),
    "cart_7d": v1.FixedLenFeature(1, tf.int64, -1),
    "wish_7d": v1.FixedLenFeature(1, tf.int64, -1),
    "pay_7d": v1.FixedLenFeature(1, tf.int64, -1),
    "pctr_7d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_7d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_7d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_7d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pv_14d": v1.FixedLenFeature(1, tf.int64, -1),
    "ipv_14d": v1.FixedLenFeature(1, tf.int64, -1),
    "cart_14d": v1.FixedLenFeature(1, tf.int64, -1),
    "wish_14d": v1.FixedLenFeature(1, tf.int64, -1),
    "pay_14d": v1.FixedLenFeature(1, tf.int64, -1),
    "pctr_14d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_14d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_14d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_14d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pv_30d": v1.FixedLenFeature(1, tf.int64, -1),
    "ipv_30d": v1.FixedLenFeature(1, tf.int64, -1),
    "cart_30d": v1.FixedLenFeature(1, tf.int64, -1),
    "wish_30d": v1.FixedLenFeature(1, tf.int64, -1),
    "pay_30d": v1.FixedLenFeature(1, tf.int64, -1),
    "pctr_30d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_30d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_30d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_30d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "mt_i2i_main_score": v1.FixedLenFeature(1, tf.float32, -1.0),
    "mt_i2i_long_score": v1.FixedLenFeature(1, tf.float32, -1.0),
    "mt_i2i_short_score": v1.FixedLenFeature(1, tf.float32, -1.0),
    "mt": v1.FixedLenFeature(6, tf.string, default_value=[""] * 6)

    , "is_rel_cate": v1.FixedLenFeature(1, tf.int64, 0)
    , "is_rel_cate2": v1.FixedLenFeature(1, tf.int64, 0)
    , "is_rel_cate3": v1.FixedLenFeature(1, tf.int64, 0)
    , "is_rel_cate4": v1.FixedLenFeature(1, tf.int64, 0)
    , "sales_price": v1.FixedLenFeature(1, tf.int64, 0)
    , "mt_i2i_main": v1.FixedLenFeature(1, tf.int64, 0)
    , "mt_i2i_long": v1.FixedLenFeature(1, tf.int64, 0)
    , "mt_i2i_short": v1.FixedLenFeature(1, tf.int64, 0)
    , "mt_hot_i2leaf": v1.FixedLenFeature(1, tf.int64, 0)
    , "mt_hot": v1.FixedLenFeature(1, tf.int64, 0)

    , "main_goods_id": v1.FixedLenFeature(1, tf.string, "-1")
    , "main_cate_id": v1.FixedLenFeature(1, tf.string, "-1")
    , "main_cate_level2_id": v1.FixedLenFeature(1, tf.string, "-1")
    , "main_cate_level3_id": v1.FixedLenFeature(1, tf.string, "-1")
    , "main_cate_level4_id": v1.FixedLenFeature(1, tf.string, "-1")

    , "prop_seaon": v1.FixedLenFeature(1, tf.string, "-1")
    , "prop_length": v1.FixedLenFeature(1, tf.string, "-1")
    , "prop_main_material": v1.FixedLenFeature(1, tf.string, "-1")
    , "prop_pattern": v1.FixedLenFeature(1, tf.string, "-1")
    , "prop_style": v1.FixedLenFeature(1, tf.string, "-1")
    , "prop_quantity": v1.FixedLenFeature(1, tf.string, "-1")
    , "prop_fitness": v1.FixedLenFeature(1, tf.string, "-1")

    , "last_login_device": v1.FixedLenFeature(1, tf.string, "-1")
    , "last_login_brand": v1.FixedLenFeature(1, tf.string, "-1")
    , "register_brand": v1.FixedLenFeature(1, tf.string, "-1")
    , "client_type": v1.FixedLenFeature(1, tf.string, "-1")

    , "cate_id": v1.FixedLenFeature(1, tf.string, "-1")
    , "goods_id": v1.FixedLenFeature(1, tf.string, "-1")
    , "cate_level2_id": v1.FixedLenFeature(1, tf.string, "-1")
    , "cate_level3_id": v1.FixedLenFeature(1, tf.string, "-1")
    , "cate_level4_id": v1.FixedLenFeature(1, tf.string, "-1")
    , "country": v1.FixedLenFeature(1, tf.string, '-1')

    , "is_clk": v1.FixedLenFeature(1, tf.int64, 0)
    , "is_pay": v1.FixedLenFeature(1, tf.int64, 0)
    , "sample_id": v1.FixedLenFeature(1, tf.string, "-1")
}