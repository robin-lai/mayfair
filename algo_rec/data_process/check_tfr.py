import os

import tensorflow as tf
import argparse
print(tf.__version__)
import tensorflow.compat.v1 as v1

def main(args):
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

            , "is_rel_cate": tf.FixedLenFeature(1, tf.int64, 0)
            , "is_rel_cate2": tf.FixedLenFeature(1, tf.int64, 0)
            , "is_rel_cate3": tf.FixedLenFeature(1, tf.int64, 0)
            , "is_rel_cate4": tf.FixedLenFeature(1, tf.int64, 0)
            , "sales_price": tf.FixedLenFeature(1, tf.int64, 0)

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
            , "highLevelSeqListGoods": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            , "highLevelSeqListCateId": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            , "lowerLevelSeqListGoods": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            , "lowerLevelSeqListCateId": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)

            , "is_clk": tf.FixedLenFeature(1, tf.int64, 0)
            , "is_pay": tf.FixedLenFeature(1, tf.int64, 0)
            , "sample_id": tf.FixedLenFeature(1, tf.string, "-1")
           }
        features = tf.io.parse_single_example(data, features=feature_describe)
        return features
    s3_file = 's3://warehouse-algo/rec/cn_rec_detail_sample_v10_tfr/ds=%s/%s' % (args.ds, args.file)
    local_file = './' + args.file
    os.system('aws s3 cp %s %s' % (s3_file, local_file))
    ds = tf.data.TFRecordDataset(local_file)
    ds2 = ds.map(parse).shuffle(args.batch_size * 10).batch(args.batch_size)
    ll = list(ds2.as_numpy_iterator())
    for idx, it in enumerate(ll):
        if args.name != '':
            for name in args.names.split(','):
                print('feature_name:', name)
                print(it[name])
        else:
            print(it)
        if idx == args.head:
            break


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
        prog='predict',
        description='predict',
        epilog='predict')
    parser.add_argument('--file', default='part-00000-17ebac5c-0e1d-4b33-98db-ebd48025b24b-c000')
    parser.add_argument('--ds', default='20241209')
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--head',type=int, default=10)
    parser.add_argument('--names', default='sample_id,highLevelSeqListGoods,lowerLevelSeqListGoods,lowerLevelSeqListCateId,is_rel_cate3,main_cate_level2_id,prop_style,last_login_device,main_goods_id')
    args = parser.parse_args()
    main(args)


