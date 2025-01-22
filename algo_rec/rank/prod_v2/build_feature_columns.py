# encoding:utf-8
import tensorflow as tf

def build_feature_columns():
    # cate-seq
    cate2_fc_emb = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key="cate_level2_id", hash_bucket_size=400),8)
    cate3_fc_emb =  tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key="cate_level3_id", hash_bucket_size=1000),8)
    cate4_fc_emb =  tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key="cate_level4_id", hash_bucket_size=2000),8)
    cate_fc_emb =   tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key="cate_id", hash_bucket_size=4000),8)
    goods_id_fc_emb =  tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key="goods_id", hash_bucket_size=200000),32)

    m_cate2_fc_emb =  tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key="main_cate_level2_id", hash_bucket_size=400),8)
    m_cate3_fc_emb =  tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key="main_cate_level3_id", hash_bucket_size=1000),8)
    m_cate4_fc_emb =  tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key="main_cate_level4_id", hash_bucket_size=2000),8)
    m_cate_fc_emb =  tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key="main_cate_id", hash_bucket_size=4000),8)
    m_goods_id_fc_emb =  tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key="main_goods_id", hash_bucket_size=200000),32)

    id_col_emb = [cate2_fc_emb,cate3_fc_emb, cate4_fc_emb,cate_fc_emb,
                  goods_id_fc_emb,m_cate2_fc_emb,m_cate3_fc_emb,m_cate4_fc_emb,
                  m_cate_fc_emb,m_goods_id_fc_emb]

    # mt_fc = tf.feature_column.categorical_column_with_vocabulary_list(
    #     key='mt', vocabulary_list=('hot', 'hot_i2leaf', 'u2i_f', 'i2i_main', 'i2i_short', 'i2i_long', ''),
    #     num_oov_buckets=2)
    # mt_emb = tf.feature_column.embedding_column(mt_fc, 16)

    # cate2_share_emb = tf.feature_column.shared_embedding_columns([cate2_fc, m_cate2_fc], dimension=16)
    # cate3_share_emb = tf.feature_column.shared_embedding_columns([cate3_fc, m_cate3_fc], dimension=16)
    # cate4_share_emb = tf.feature_column.shared_embedding_columns([cate4_fc, m_cate4_fc], dimension=16)
    # cate_share_emb = tf.feature_column.shared_embedding_columns([cate_fc, m_cate_fc], dimension=16)
    # goods_share_emb = tf.feature_column.shared_embedding_columns([goods_id_fc, m_goods_id_fc], dimension=32)
    # cate1_emb = tf.feature_column.embedding_column(cate1_fc, 16)

    # cate_cols_share_emb = [cate2_share_emb, cate3_share_emb, cate4_share_emb, cate_share_emb,
    #                        goods_share_emb]

    prop_seaon_fc = tf.feature_column.categorical_column_with_hash_bucket(key="prop_seaon", hash_bucket_size=30)
    prop_length_fc = tf.feature_column.categorical_column_with_hash_bucket(key="prop_length", hash_bucket_size=30)
    prop_main_material_fc = tf.feature_column.categorical_column_with_hash_bucket(key="prop_main_material",
                                                                                  hash_bucket_size=30)
    prop_pattern_fc = tf.feature_column.categorical_column_with_hash_bucket(key="prop_pattern", hash_bucket_size=100)
    prop_style_fc = tf.feature_column.categorical_column_with_hash_bucket(key="prop_style", hash_bucket_size=200)
    prop_quantity_fc = tf.feature_column.categorical_column_with_hash_bucket(key="prop_quantity", hash_bucket_size=200)
    prop_fitness_fc = tf.feature_column.categorical_column_with_hash_bucket(key="prop_fitness", hash_bucket_size=200)
    last_login_device_fc = tf.feature_column.categorical_column_with_hash_bucket(key="last_login_device",
                                                                                 hash_bucket_size=2000)
    last_login_brand_fc = tf.feature_column.categorical_column_with_hash_bucket(key="last_login_brand",
                                                                                hash_bucket_size=1000)
    register_brand_fc = tf.feature_column.categorical_column_with_hash_bucket(key="register_brand",
                                                                              hash_bucket_size=1000)

    prop_seaon_emb = tf.feature_column.embedding_column(prop_seaon_fc, 8)
    prop_length_emb = tf.feature_column.embedding_column(prop_length_fc, 8)
    prop_main_material_emb = tf.feature_column.embedding_column(prop_main_material_fc, 8)
    prop_pattern_emb = tf.feature_column.embedding_column(prop_pattern_fc, 8)
    prop_style_emb = tf.feature_column.embedding_column(prop_style_fc, 8)
    prop_quantity_emb = tf.feature_column.embedding_column(prop_quantity_fc, 8)
    prop_fitness_emb = tf.feature_column.embedding_column(prop_fitness_fc, 8)

    last_login_device_emb = tf.feature_column.embedding_column(last_login_device_fc, 8)
    last_login_brand_emb = tf.feature_column.embedding_column(last_login_brand_fc, 8)
    register_brand_emb = tf.feature_column.embedding_column(register_brand_fc, 8)

    cate_cols_emb = [prop_seaon_emb, prop_length_emb, prop_main_material_emb, prop_pattern_emb, prop_style_emb,
                     prop_quantity_emb, prop_fitness_emb
        , last_login_device_emb, last_login_brand_emb, register_brand_emb,mt_emb]

    # int col
    is_rel_cate_fc = tf.feature_column.categorical_column_with_identity("is_rel_cate", num_buckets=3, default_value=0)
    is_rel_cate2_fc = tf.feature_column.categorical_column_with_identity("is_rel_cate2", num_buckets=3, default_value=0)
    is_rel_cate3_fc = tf.feature_column.categorical_column_with_identity("is_rel_cate3", num_buckets=3, default_value=0)
    is_rel_cate4_fc = tf.feature_column.categorical_column_with_identity("is_rel_cate4", num_buckets=3, default_value=0)
    sales_price_fc = tf.feature_column.categorical_column_with_identity("sales_price", num_buckets=20, default_value=0)
    mt_i2i_main_emb = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_identity("mt_i2i_main", num_buckets=3, default_value=0), 8)
    mt_i2i_main_score = tf.feature_column.numeric_column(key="mt_i2i_main_score", default_value=-1, dtype=tf.float32)
    mt_i2i_long_emb = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_identity("mt_i2i_long", num_buckets=3, default_value=0), 8)
    mt_i2i_long_score = tf.feature_column.numeric_column(key="mt_i2i_long_score", default_value=-1, dtype=tf.float32)
    mt_i2i_short_emb = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_identity("mt_i2i_short", num_buckets=3, default_value=0), 8)
    mt_i2i_short_score = tf.feature_column.numeric_column(key="mt_i2i_short_score", default_value=-1, dtype=tf.float32)
    mt_hot_i2leaf_emb = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_identity("mt_hot_i2leaf", num_buckets=3, default_value=0), 8)
    mt_hot_emb = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_identity("mt_hot", num_buckets=3, default_value=0), 8)


    is_rel_cate_emb = tf.feature_column.embedding_column(is_rel_cate_fc, 8)
    is_rel_cate2_emb = tf.feature_column.embedding_column(is_rel_cate2_fc, 8)
    is_rel_cate3_emb = tf.feature_column.embedding_column(is_rel_cate3_fc, 8)
    is_rel_cate4_emb = tf.feature_column.embedding_column(is_rel_cate4_fc, 8)
    sales_price_emb = tf.feature_column.embedding_column(sales_price_fc, 8)

    numric_cols_emb = [is_rel_cate_emb, is_rel_cate2_emb, is_rel_cate3_emb, is_rel_cate4_emb, sales_price_emb,
                       mt_i2i_main_emb,mt_i2i_main_score,mt_i2i_long_emb,mt_i2i_long_score,mt_i2i_short_emb,
                       mt_i2i_short_score,mt_hot_i2leaf_emb,mt_hot_emb]

    pv_1d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pv_1d"),
                                            boundaries=[337.0, 380.0, 427.0, 481.0, 607.0, 691.0, 779.0, 872.0, 985.0,
                                                        100025.0]), 4)
    ipv_1d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="ipv_1d"),
                                            boundaries=[38.0, 45.0, 52.0, 58.0, 74.0, 83.0, 94.0, 106.0, 120.0,
                                                        6191.0]), 4)
    cart_1d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="cart_1d"),
                                            boundaries=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 339.0]), 4)
    wish_1d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="wish_1d"),
                                            boundaries=[0.0, 1.0, 2.0, 3.0, 4.0, 491.0]), 4)
    pay_1d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pay_1d"), boundaries=[0.0, 37.0]), 4)
    pctr_1d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pctr_1d"),
                                            boundaries=[0.08216, 0.09079, 0.09581, 0.0999, 0.10664, 0.10943, 0.11204,
                                                        0.11478, 0.11747, 0.30088]), 4)
    pcart_1d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pcart_1d"),
                                            boundaries=[0.00699, 0.01786, 0.02439, 0.03073, 0.04082, 0.04602, 0.05128,
                                                        0.05634, 0.06122, 0.35484]), 4)
    pwish_1d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pwish_1d"),
                                            boundaries=[0.0, 0.00446, 0.00795, 0.01493, 0.02857, 0.04274, 0.40541]), 4)
    pcvr_1d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pcvr_1d"), boundaries=[0.0, 0.16832]),
        4)
    pv_3d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pv_3d"),
                                            boundaries=[369.0, 452.0, 545.0, 664.0, 962.0, 1171.0, 1415.0, 1741.0,
                                                        2089.0, 2530.0, 3102.0, 3809.0, 4715.0, 5897.0, 7572.0, 9898.0,
                                                        13620.0, 538767.0]), 4)
    ipv_3d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="ipv_3d"),
                                            boundaries=[43.0, 54.0, 66.0, 80.0, 118.0, 143.0, 173.0, 209.0, 252.0,
                                                        305.0, 369.0, 458.0, 564.0, 702.0, 884.0, 1159.0, 1562.0,
                                                        34066.0]), 4)
    cart_3d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="cart_3d"),
                                            boundaries=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 23.0,
                                                        29.0, 36.0, 47.0, 61.0, 82.0, 117.0, 2221.0]), 4)
    wish_3d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="wish_3d"),
                                            boundaries=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 15.0, 20.0, 27.0,
                                                        38.0, 55.0, 83.0, 2256.0]), 4)
    pay_3d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pay_3d"),
                                            boundaries=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 218.0]), 4)
    pctr_3d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pctr_3d"),
                                            boundaries=[0.08038, 0.0905, 0.09648, 0.10089, 0.10766, 0.11051, 0.11357,
                                                        0.11642, 0.11929, 0.12218, 0.1254, 0.12844, 0.13202, 0.13592,
                                                        0.14058, 0.14641, 0.15399, 0.30952]), 4)
    pcart_3d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pcart_3d"),
                                            boundaries=[0.01064, 0.01942, 0.02564, 0.03125, 0.0411, 0.0456, 0.05,
                                                        0.05418, 0.05869, 0.06318, 0.06852, 0.07407, 0.08016, 0.0875,
                                                        0.09554, 0.10667, 0.12213, 0.39286]), 4)
    pwish_3d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pwish_3d"),
                                            boundaries=[0.0, 0.0024, 0.0057, 0.00882, 0.02239, 0.04, 0.05072, 0.05957,
                                                        0.06722, 0.07447, 0.08117, 0.0883, 0.09623, 0.10638, 0.11765,
                                                        0.30189]), 4)
    pcvr_3d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pcvr_3d"),
                                            boundaries=[0.0, 0.00113, 0.00229, 0.00325, 0.00426, 0.00529, 0.00643,
                                                        0.00787, 0.00962, 0.0119, 0.01566, 0.20859]), 4)
    pv_5d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pv_5d"),
                                            boundaries=[390.0, 499.0, 617.0, 757.0, 1143.0, 1393.0, 1703.0, 2102.0,
                                                        2575.0, 3182.0, 3918.0, 4892.0, 6118.0, 7832.0, 10140.0,
                                                        786240.0]), 4)
    ipv_5d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="ipv_5d"),
                                            boundaries=[46.0, 59.0, 74.0, 92.0, 139.0, 170.0, 209.0, 256.0, 315.0,
                                                        384.0, 475.0, 588.0, 739.0, 936.0, 1198.0, 49431.0]), 4)
    cart_5d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="cart_5d"),
                                            boundaries=[1.0, 2.0, 3.0, 4.0, 7.0, 9.0, 12.0, 15.0, 19.0, 23.0, 29.0,
                                                        37.0, 47.0, 62.0, 82.0, 2437.0]), 4)
    wish_5d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="wish_5d"),
                                            boundaries=[0.0, 1.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 14.0, 19.0, 25.0, 35.0,
                                                        50.0, 3125.0]), 4)
    pay_5d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pay_5d"),
                                            boundaries=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 228.0]), 4)
    pctr_5d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pctr_5d"),
                                            boundaries=[0.08089, 0.09091, 0.09722, 0.10175, 0.10841, 0.11123, 0.11437,
                                                        0.11729, 0.12023, 0.12299, 0.12609, 0.12924, 0.13272, 0.13672,
                                                        0.14141, 0.29799]), 4)
    pcart_5d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pcart_5d"),
                                            boundaries=[0.01205, 0.02062, 0.02676, 0.03226, 0.04167, 0.04595, 0.05017,
                                                        0.05446, 0.05882, 0.06338, 0.06849, 0.07397, 0.08, 0.08696,
                                                        0.09544, 0.39286]), 4)
    pwish_5d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pwish_5d"),
                                            boundaries=[0.0, 0.0013, 0.00281, 0.00585, 0.00943, 0.025, 0.04187, 0.05263,
                                                        0.06122, 0.06849, 0.07527, 0.08193, 0.08929, 0.09756, 0.27165]),
        4)
    pcvr_5d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pcvr_5d"),
                                            boundaries=[0.0, 0.00061, 0.00192, 0.00275, 0.00361, 0.00455, 0.00551,
                                                        0.00658, 0.00779, 0.00935, 0.23188]), 4)
    pv_7d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pv_7d"),
                                            boundaries=[391.0, 517.0, 661.0, 826.0, 1266.0, 1583.0, 1948.0, 2411.0,
                                                        3011.0, 3710.0, 4615.0, 5857.0, 7420.0, 9575.0, 1063995.0]), 4)
    ipv_7d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="ipv_7d"),
                                            boundaries=[46.0, 62.0, 80.0, 100.0, 157.0, 193.0, 238.0, 295.0, 368.0,
                                                        457.0, 565.0, 707.0, 901.0, 1171.0, 67171.0]), 4)
    cart_7d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="cart_7d"),
                                            boundaries=[1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 13.0, 17.0, 21.0, 27.0, 35.0,
                                                        44.0, 58.0, 76.0, 3385.0]), 4)
    wish_7d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="wish_7d"),
                                            boundaries=[0.0, 1.0, 3.0, 5.0, 6.0, 8.0, 10.0, 13.0, 17.0, 22.0, 30.0,
                                                        42.0, 4186.0]), 4)
    pay_7d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pay_7d"),
                                            boundaries=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 285.0]), 4)
    pctr_7d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pctr_7d"),
                                            boundaries=[0.08091, 0.09143, 0.09758, 0.10225, 0.10921, 0.11215, 0.1151,
                                                        0.11798, 0.12084, 0.12387, 0.12686, 0.13012, 0.13363, 0.13749,
                                                        0.29799]), 4)
    pcart_7d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pcart_7d"),
                                            boundaries=[0.01235, 0.02083, 0.02728, 0.03262, 0.04185, 0.04615, 0.05033,
                                                        0.05439, 0.05882, 0.06329, 0.06826, 0.07353, 0.0792, 0.08599,
                                                        0.44872]), 4)
    pwish_7d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pwish_7d"),
                                            boundaries=[0.0, 0.00174, 0.00303, 0.00602, 0.00965, 0.02632, 0.04286,
                                                        0.05263, 0.06091, 0.06852, 0.07534, 0.08197, 0.08929, 0.35714]),
        4)
    pcvr_7d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pcvr_7d"),
                                            boundaries=[0.0, 0.00144, 0.00236, 0.00315, 0.00401, 0.00485, 0.00574,
                                                        0.00676, 0.00806, 0.19048]), 4)
    pv_14d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pv_14d"),
                                            boundaries=[439.0, 601.0, 825.0, 1094.0, 1798.0, 2258.0, 2857.0, 3571.0,
                                                        4489.0, 5734.0, 7281.0, 9341.0, 12279.0, 16178.0, 21386.0,
                                                        29854.0, 43283.0, 70605.0, 2165811.0]), 4)
    ipv_14d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="ipv_14d"),
                                            boundaries=[48.0, 71.0, 97.0, 129.0, 214.0, 273.0, 345.0, 435.0, 554.0,
                                                        705.0, 900.0, 1146.0, 1495.0, 1963.0, 2595.0, 3592.0, 5059.0,
                                                        8144.0, 139822.0]), 4)
    cart_14d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="cart_14d"),
                                            boundaries=[1.0, 3.0, 4.0, 6.0, 11.0, 14.0, 18.0, 24.0, 31.0, 41.0, 55.0,
                                                        71.0, 95.0, 129.0, 175.0, 252.0, 367.0, 642.0, 7056.0]), 4)
    wish_14d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="wish_14d"),
                                            boundaries=[0.0, 1.0, 2.0, 5.0, 7.0, 9.0, 12.0, 15.0, 20.0, 27.0, 35.0,
                                                        48.0, 69.0, 101.0, 157.0, 258.0, 483.0, 8056.0]), 4)
    pay_14d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pay_14d"),
                                            boundaries=[0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 11.0, 16.0, 23.0, 36.0, 62.0,
                                                        667.0]), 4)
    pctr_14d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pctr_14d"),
                                            boundaries=[0.07635, 0.08911, 0.09642, 0.10149, 0.10904, 0.11221, 0.11518,
                                                        0.11804, 0.12101, 0.12406, 0.12708, 0.13028, 0.13396, 0.13793,
                                                        0.14235, 0.14807, 0.15591, 0.16858, 0.29799]), 4)
    pcart_14d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pcart_14d"),
                                            boundaries=[0.01385, 0.02206, 0.02819, 0.03311, 0.04204, 0.0461, 0.04986,
                                                        0.05405, 0.05825, 0.0625, 0.06716, 0.07201, 0.07756, 0.08407,
                                                        0.09217, 0.10211, 0.11596, 0.14025, 0.3675]), 4)
    pwish_14d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pwish_14d"),
                                            boundaries=[0.0, 0.00248, 0.00356, 0.00638, 0.01029, 0.02936, 0.04425,
                                                        0.05362, 0.06122, 0.06796, 0.07512, 0.08133, 0.08824, 0.09604,
                                                        0.10507, 0.11749, 0.13575, 0.37097]), 4)
    pcvr_14d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pcvr_14d"),
                                            boundaries=[0.0, 0.00151, 0.00237, 0.0031, 0.00386, 0.00463, 0.00541,
                                                        0.00627, 0.00727, 0.0084, 0.00971, 0.01153, 0.01447, 0.02012,
                                                        0.18519]), 4)
    pv_30d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pv_30d"),
                                            boundaries=[468.0, 698.0, 999.0, 1388.0, 2504.0, 3273.0, 4182.0, 5358.0,
                                                        6873.0, 8817.0, 11591.0, 15376.0, 20689.0, 28140.0, 38351.0,
                                                        53810.0, 81805.0, 137372.0, 4691307.0]), 4)
    ipv_30d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="ipv_30d"),
                                            boundaries=[48.0, 78.0, 115.0, 162.0, 297.0, 391.0, 505.0, 654.0, 850.0,
                                                        1105.0, 1459.0, 1930.0, 2586.0, 3476.0, 4702.0, 6564.0, 9864.0,
                                                        16383.0, 319416.0]), 4)
    cart_30d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="cart_30d"),
                                            boundaries=[1.0, 3.0, 5.0, 8.0, 15.0, 20.0, 27.0, 36.0, 48.0, 65.0, 89.0,
                                                        121.0, 164.0, 228.0, 322.0, 466.0, 715.0, 1282.0, 14776.0]), 4)
    wish_30d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="wish_30d"),
                                            boundaries=[0.0, 1.0, 2.0, 3.0, 7.0, 10.0, 14.0, 19.0, 25.0, 34.0, 45.0,
                                                        61.0, 83.0, 119.0, 176.0, 281.0, 468.0, 908.0, 13197.0]), 4)
    pay_30d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pay_30d"),
                                            boundaries=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 15.0, 21.0, 31.0, 45.0,
                                                        72.0, 133.0, 2077.0]), 4)
    pctr_30d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pctr_30d"),
                                            boundaries=[0.06979, 0.08509, 0.094, 0.10049, 0.10888, 0.11238, 0.11565,
                                                        0.11867, 0.12179, 0.12496, 0.12815, 0.13154, 0.13507, 0.13926,
                                                        0.14388, 0.14963, 0.15745, 0.17, 0.30357]), 4)
    pcart_30d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pcart_30d"),
                                            boundaries=[0.01394, 0.02273, 0.02857, 0.03358, 0.04237, 0.04615, 0.05006,
                                                        0.05423, 0.0581, 0.06234, 0.06667, 0.07143, 0.07692, 0.08305,
                                                        0.09065, 0.10045, 0.11462, 0.13842, 0.35294]), 4)
    pwish_30d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pwish_30d"),
                                            boundaries=[0.0, 0.00175, 0.00326, 0.00436, 0.00714, 0.01136, 0.03217,
                                                        0.04461, 0.05319, 0.06091, 0.06716, 0.07348, 0.07949, 0.08621,
                                                        0.09368, 0.10249, 0.11429, 0.13285, 0.33333]), 4)
    pcvr_30d_emb = tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pcvr_30d"),
                                            boundaries=[0.0, 0.00144, 0.00225, 0.00299, 0.0037, 0.00439, 0.00511,
                                                        0.00588, 0.00671, 0.00761, 0.00864, 0.0099, 0.01166, 0.01452,
                                                        0.0198, 0.21429]), 4)

    numric_cols_emb.extend(
        [pv_1d_emb, ipv_1d_emb, cart_1d_emb, wish_1d_emb, pay_1d_emb, pctr_1d_emb, pcart_1d_emb, pwish_1d_emb,
         pcvr_1d_emb, pv_3d_emb, ipv_3d_emb, cart_3d_emb, wish_3d_emb, pay_3d_emb, pctr_3d_emb, pcart_3d_emb,
         pwish_3d_emb, pcvr_3d_emb, pv_5d_emb, ipv_5d_emb, cart_5d_emb, wish_5d_emb, pay_5d_emb, pctr_5d_emb,
         pcart_5d_emb, pwish_5d_emb, pcvr_5d_emb, pv_7d_emb, ipv_7d_emb, cart_7d_emb, wish_7d_emb, pay_7d_emb,
         pctr_7d_emb, pcart_7d_emb, pwish_7d_emb, pcvr_7d_emb, pv_14d_emb, ipv_14d_emb, cart_14d_emb,
         wish_14d_emb, pay_14d_emb, pctr_14d_emb, pcart_14d_emb, pwish_14d_emb, pcvr_14d_emb, pv_30d_emb,
         ipv_30d_emb, cart_30d_emb, wish_30d_emb, pay_30d_emb, pctr_30d_emb, pcart_30d_emb, pwish_30d_emb,
         pcvr_30d_emb])

    return {"cate_cols_emb": cate_cols_emb, "numric_cols_emb": numric_cols_emb,
            # "cate_cols_share_emb": cate_cols_share_emb
            "id_col_emb": id_col_emb
            }