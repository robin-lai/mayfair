import tensorflow as tf
import argparse
print(tf.__version__)

def main(args):
    # import tensorflow.compat.v1 as v1

    def parse(data):
        feature_describe = {
            # "ctr_7d": v1.FixedLenFeature(1, tf.float32, 0.0)
            # , "cvr_7d": v1.FixedLenFeature(1, tf.float32, 0.0)
            # , "show_7d": v1.FixedLenFeature(1, tf.int64, 0)
            # , "click_7d": v1.FixedLenFeature(1, tf.int64, 0)
            # , "cart_7d": v1.FixedLenFeature(1, tf.int64, 0)
            # , "ord_total": v1.FixedLenFeature(1, tf.int64, 0)
            # , "pay_total": v1.FixedLenFeature(1, tf.int64, 0)
            # , "ord_7d": v1.FixedLenFeature(1, tf.int64, 0)
            # , "pay_7d": v1.FixedLenFeature(1, tf.int64, 0)
            #
            # , "is_rel_cate": v1.FixedLenFeature(1, tf.int64, 0)
            # , "is_rel_cate2": v1.FixedLenFeature(1, tf.int64, 0)
            # , "is_rel_cate3": v1.FixedLenFeature(1, tf.int64, 0)
            # , "is_rel_cate4": v1.FixedLenFeature(1, tf.int64, 0)
            # , "sales_price": v1.FixedLenFeature(1, tf.int64, 0)
            #
            # , "main_goods_id": v1.FixedLenFeature(1, tf.string, "-1")
            # , "main_cate_id": v1.FixedLenFeature(1, tf.string, "-1")
             "main_cate_level2_id": v1.FixedLenFeature(1, tf.string, "-1")
            # , "main_cate_level3_id": v1.FixedLenFeature(1, tf.string, "-1")
            # , "main_cate_level4_id": v1.FixedLenFeature(1, tf.string, "-1")
            #
            # , "prop_seaon": v1.FixedLenFeature(1, tf.string, "-1")
            # , "prop_length": v1.FixedLenFeature(1, tf.string, "-1")
            # , "prop_main_material": v1.FixedLenFeature(1, tf.string, "-1")
            # , "prop_pattern": v1.FixedLenFeature(1, tf.string, "-1")
            # , "prop_style": v1.FixedLenFeature(1, tf.string, "-1")
            # , "prop_quantity": v1.FixedLenFeature(1, tf.string, "-1")
            # , "prop_fitness": v1.FixedLenFeature(1, tf.string, "-1")
            #
            # , "last_login_device": v1.FixedLenFeature(1, tf.string, "-1")
            # , "last_login_brand": v1.FixedLenFeature(1, tf.string, "-1")
            # , "register_brand": v1.FixedLenFeature(1, tf.string, "-1")
            # # , "client_type": v1.FixedLenFeature(1, tf.string, "-1")
            #
            # , "cate_id": v1.FixedLenFeature(1, tf.string, "-1")
            # , "goods_id": v1.FixedLenFeature(1, tf.string, "-1")
            # , "cate_level1_id": v1.FixedLenFeature(1, tf.string, "-1")
            # , "cate_level2_id": v1.FixedLenFeature(1, tf.string, "-1")
            # , "cate_level3_id": v1.FixedLenFeature(1, tf.string, "-1")
            # , "cate_level4_id": v1.FixedLenFeature(1, tf.string, "-1")
            # , "country": v1.FixedLenFeature(1, tf.string, '-1')

            # , "seq_cate_id": v1.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
            # , "seq_goods_id": v1.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
            # , "seq_cate_id": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            # , "seq_goods_id": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            # , "highLevelSeqListGoods": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            , "highLevelSeqListCateId": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            # , "lowerLevelSeqListGoods": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            , "lowerLevelSeqListCateId": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20),
            "highLevelSeqList_len": v1.FixedLenFeature(1, tf.int64, default_value=0),
            "lowerLevelSeqList_len": v1.FixedLenFeature(1, tf.int64, default_value=0),
            # "is_clk": v1.FixedLenFeature(1, tf.int64, 0)
            # , "is_pay": v1.FixedLenFeature(1, tf.int64, 0)
            # , "sample_id": v1.FixedLenFeature(1, tf.string, "-1")
        }
        features = tf.io.parse_single_example(data, features=feature_describe)
        return features

    ds = tf.data.TFRecordDataset(args.file)
    ds = ds.map(parse).batch(100)
    print(list(ds.as_numpy_iterator())[0:args.n])
    # [{'is_clk': array([[1],
    #          [1]]),
    #   'is_pay': array([[0],
    #          [0]])},
    #  {'is_clk': array([[1],
    #          [1]]),
    #   'is_pay': array([[0],
    #          [0]])},

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gentfr',
        description='gentfr',
        epilog='gentfr-help')
    parser.add_argument('--file',type=str, default='./tfrecord/part-00000-18b4c5ae-0eba-41d2-b246-79e7f457ee3d-c000')
    parser.add_argument('--n',type=int, default=10)
    args = parser.parse_args()
    main(args)



