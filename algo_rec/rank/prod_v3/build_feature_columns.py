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
        [pctr_14d_emb, pcart_14d_emb, pwish_14d_emb, pcvr_14d_emb,
         pctr_30d_emb, pcart_30d_emb, pwish_30d_emb,
         pcvr_30d_emb])

    return {"numric_cols_emb": numric_cols_emb,
            "id_col_emb": id_col_emb
            }