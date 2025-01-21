# encoding:utf-8
import tensorflow as tf

feature_spec_serve = {
    # bigtint 类型字段
    "pv_1d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pv_1d"),
    "ipv_1d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="ipv_1d"),
    "cart_1d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="cart_1d"),
    "wish_1d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="wish_1d"),
    "pay_1d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pay_1d"),
    "pv_3d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pv_3d"),
    "ipv_3d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="ipv_3d"),
    "cart_3d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="cart_3d"),
    "wish_3d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="wish_3d"),
    "pay_3d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pay_3d"),
    "pv_5d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pv_5d"),
    "ipv_5d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="ipv_5d"),
    "cart_5d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="cart_5d"),
    "wish_5d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="wish_5d"),
    "pay_5d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pay_5d"),
    "pv_7d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pv_7d"),
    "ipv_7d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="ipv_7d"),
    "cart_7d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="cart_7d"),
    "wish_7d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="wish_7d"),
    "pay_7d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pay_7d"),
    "pv_14d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pv_14d"),
    "ipv_14d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="ipv_14d"),
    "cart_14d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="cart_14d"),
    "wish_14d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="wish_14d"),
    "pay_14d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pay_14d"),
    "pv_30d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pv_30d"),
    "ipv_30d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="ipv_30d"),
    "cart_30d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="cart_30d"),
    "wish_30d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="wish_30d"),
    "pay_30d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pay_30d"),

    # double 类型字段
    "pctr_1d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pctr_1d"),
    "pcart_1d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pcart_1d"),
    "pwish_1d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pwish_1d"),
    "pcvr_1d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pcvr_1d"),
    "pctr_3d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pctr_3d"),
    "pcart_3d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pcart_3d"),
    "pwish_3d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pwish_3d"),
    "pcvr_3d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pcvr_3d"),
    "pctr_5d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pctr_5d"),
    "pcart_5d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pcart_5d"),
    "pwish_5d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pwish_5d"),
    "pcvr_5d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pcvr_5d"),
    "pctr_7d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pctr_7d"),
    "pcart_7d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pcart_7d"),
    "pwish_7d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pwish_7d"),
    "pcvr_7d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="pcvr_7d"),
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

    "is_rel_cate": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="is_rel_cate"),
    "is_rel_cate2": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="is_rel_cate2"),
    "is_rel_cate3": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="is_rel_cate3"),
    "is_rel_cate4": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="is_rel_cate4"),
    "sales_price": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="sales_price"),

    "main_goods_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="main_goods_id"),
    "main_cate_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="main_cate_id"),
    "main_cate_level2_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="main_cate_level2_id"),
    "main_cate_level3_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="main_cate_level3_id"),
    "main_cate_level4_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="main_cate_level4_id"),

    "prop_seaon": tf.placeholder(dtype=tf.string, shape=[None, 1], name="prop_seaon"),
    "prop_length": tf.placeholder(dtype=tf.string, shape=[None, 1], name="prop_length"),
    "prop_main_material": tf.placeholder(dtype=tf.string, shape=[None, 1], name="prop_main_material"),
    "prop_pattern": tf.placeholder(dtype=tf.string, shape=[None, 1], name="prop_pattern"),
    "prop_style": tf.placeholder(dtype=tf.string, shape=[None, 1], name="prop_style"),
    "prop_quantity": tf.placeholder(dtype=tf.string, shape=[None, 1], name="prop_quantity"),
    "prop_fitness": tf.placeholder(dtype=tf.string, shape=[None, 1], name="prop_fitness"),

    "last_login_device": tf.placeholder(dtype=tf.string, shape=[None, 1], name="last_login_device"),
    "last_login_brand": tf.placeholder(dtype=tf.string, shape=[None, 1], name="last_login_brand"),
    "register_brand": tf.placeholder(dtype=tf.string, shape=[None, 1], name="register_brand"),
    "client_type": tf.placeholder(dtype=tf.string, shape=[None, 1], name="client_type"),

    "goods_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="goods_id"),
    "cate_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_id"),
    "cate_level1_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_level1_id"),
    "cate_level2_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_level2_id"),
    "cate_level3_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_level3_id"),
    "cate_level4_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_level4_id"),
    "country": tf.placeholder(dtype=tf.string, shape=[None, 1], name="country"),
    # "seq_cate_id": tf.placeholder(dtype=tf.string, shape=[None, 20], name="seq_cate_id"),
    # "seq_goods_id": tf.placeholder(dtype=tf.string, shape=[None, 20], name="seq_goods_id"),
    # "highLevelSeqListGoods": tf.placeholder(dtype=tf.string, shape=[None, 20], name="seq_goods_id"),
    # "highLevelSeqListCateId": tf.placeholder(dtype=tf.string, shape=[None, 20], name="seq_goods_id"),
    # "lowerLevelSeqListGoods": tf.placeholder(dtype=tf.string, shape=[None, 20], name="seq_goods_id"),
    # "lowerLevelSeqListCateId": tf.placeholder(dtype=tf.string, shape=[None, 20], name="seq_goods_id")
}

feature_describe = {
    "pv_1d": tf.FixedLenFeature(1, tf.int64, -1),
    "ipv_1d": tf.FixedLenFeature(1, tf.int64, -1),
    "cart_1d": tf.FixedLenFeature(1, tf.int64, -1),
    "wish_1d": tf.FixedLenFeature(1, tf.int64, -1),
    "pay_1d": tf.FixedLenFeature(1, tf.int64, -1),
    "pctr_1d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_1d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_1d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_1d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pv_3d": tf.FixedLenFeature(1, tf.int64, -1),
    "ipv_3d": tf.FixedLenFeature(1, tf.int64, -1),
    "cart_3d": tf.FixedLenFeature(1, tf.int64, -1),
    "wish_3d": tf.FixedLenFeature(1, tf.int64, -1),
    "pay_3d": tf.FixedLenFeature(1, tf.int64, -1),
    "pctr_3d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_3d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_3d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_3d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pv_5d": tf.FixedLenFeature(1, tf.int64, -1),
    "ipv_5d": tf.FixedLenFeature(1, tf.int64, -1),
    "cart_5d": tf.FixedLenFeature(1, tf.int64, -1),
    "wish_5d": tf.FixedLenFeature(1, tf.int64, -1),
    "pay_5d": tf.FixedLenFeature(1, tf.int64, -1),
    "pctr_5d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_5d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_5d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_5d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pv_7d": tf.FixedLenFeature(1, tf.int64, -1),
    "ipv_7d": tf.FixedLenFeature(1, tf.int64, -1),
    "cart_7d": tf.FixedLenFeature(1, tf.int64, -1),
    "wish_7d": tf.FixedLenFeature(1, tf.int64, -1),
    "pay_7d": tf.FixedLenFeature(1, tf.int64, -1),
    "pctr_7d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_7d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_7d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_7d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pv_14d": tf.FixedLenFeature(1, tf.int64, -1),
    "ipv_14d": tf.FixedLenFeature(1, tf.int64, -1),
    "cart_14d": tf.FixedLenFeature(1, tf.int64, -1),
    "wish_14d": tf.FixedLenFeature(1, tf.int64, -1),
    "pay_14d": tf.FixedLenFeature(1, tf.int64, -1),
    "pctr_14d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_14d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_14d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_14d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pv_30d": tf.FixedLenFeature(1, tf.int64, -1),
    "ipv_30d": tf.FixedLenFeature(1, tf.int64, -1),
    "cart_30d": tf.FixedLenFeature(1, tf.int64, -1),
    "wish_30d": tf.FixedLenFeature(1, tf.int64, -1),
    "pay_30d": tf.FixedLenFeature(1, tf.int64, -1),
    "pctr_30d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_30d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_30d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_30d": tf.FixedLenFeature(1, tf.float32, -1.0),
    "mt_i2i_main_score": tf.FixedLenFeature(1, tf.float32, -1.0),
    "mt_i2i_long_score": tf.FixedLenFeature(1, tf.float32, -1.0),
    "mt_i2i_short_score": tf.FixedLenFeature(1, tf.float32, -1.0)

    , "is_rel_cate": tf.FixedLenFeature(1, tf.int64, 0)
    , "is_rel_cate2": tf.FixedLenFeature(1, tf.int64, 0)
    , "is_rel_cate3": tf.FixedLenFeature(1, tf.int64, 0)
    , "is_rel_cate4": tf.FixedLenFeature(1, tf.int64, 0)
    , "sales_price": tf.FixedLenFeature(1, tf.int64, 0)
    , "mt_i2i_main": tf.FixedLenFeature(1, tf.int64, 0)
    , "mt_i2i_long": tf.FixedLenFeature(1, tf.int64, 0)
    , "mt_i2i_short": tf.FixedLenFeature(1, tf.int64, 0)
    , "mt_hot_i2leaf": tf.FixedLenFeature(1, tf.int64, 0)
    , "mt_hot": tf.FixedLenFeature(1, tf.int64, 0)

    , "main_goods_id": tf.FixedLenFeature(1, tf.string, "-1")
    , "main_cate_id": tf.FixedLenFeature(1, tf.string, "-1")
    , "main_cate_level2_id": tf.FixedLenFeature(1, tf.string, "-1")
    , "main_cate_level3_id": tf.FixedLenFeature(1, tf.string, "-1")
    , "main_cate_level4_id": tf.FixedLenFeature(1, tf.string, "-1")

    , "prop_seaon": tf.FixedLenFeature(1, tf.string, "-1")
    , "prop_length": tf.FixedLenFeature(1, tf.string, "-1")
    , "prop_main_material": tf.FixedLenFeature(1, tf.string, "-1")
    , "prop_pattern": tf.FixedLenFeature(1, tf.string, "-1")
    , "prop_style": tf.FixedLenFeature(1, tf.string, "-1")
    , "prop_quantity": tf.FixedLenFeature(1, tf.string, "-1")
    , "prop_fitness": tf.FixedLenFeature(1, tf.string, "-1")

    , "last_login_device": tf.FixedLenFeature(1, tf.string, "-1")
    , "last_login_brand": tf.FixedLenFeature(1, tf.string, "-1")
    , "register_brand": tf.FixedLenFeature(1, tf.string, "-1")
    , "client_type": tf.FixedLenFeature(1, tf.string, "-1")

    , "cate_id": tf.FixedLenFeature(1, tf.string, "-1")
    , "goods_id": tf.FixedLenFeature(1, tf.string, "-1")
    , "cate_level1_id": tf.FixedLenFeature(1, tf.string, "-1")
    , "cate_level2_id": tf.FixedLenFeature(1, tf.string, "-1")
    , "cate_level3_id": tf.FixedLenFeature(1, tf.string, "-1")
    , "cate_level4_id": tf.FixedLenFeature(1, tf.string, "-1")
    , "country": tf.FixedLenFeature(1, tf.string, '-1')

    # , "seq_cate_id": tf.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
    # , "seq_goods_id": tf.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
    # , "seq_cate_id": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)
    # , "seq_goods_id": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)
    # , "highLevelSeqListGoods": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)
    # , "highLevelSeqListCateId": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)
    # , "lowerLevelSeqListGoods": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)
    # , "lowerLevelSeqListCateId": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)

    , "is_clk": tf.FixedLenFeature(1, tf.int64, 0)
    , "is_pay": tf.FixedLenFeature(1, tf.int64, 0)
    , "sample_id": tf.FixedLenFeature(1, tf.string, "-1")
}