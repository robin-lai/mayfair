import os

import tensorflow as tf
import argparse
print(tf.__version__)
import tensorflow.compat.v1 as v1

def main(args):
    def parse(data):
        feature_describe = {
            "ctr_7d": v1.FixedLenFeature(1, tf.float32, 0.0)
            , "cvr_7d": v1.FixedLenFeature(1, tf.float32, 0.0)
            , "show_7d": v1.FixedLenFeature(1, tf.int64, 0)
            , "click_7d": v1.FixedLenFeature(1, tf.int64, 0)
            , "cart_7d": v1.FixedLenFeature(1, tf.int64, 0)
            , "ord_total": v1.FixedLenFeature(1, tf.int64, 0)
            , "pay_total": v1.FixedLenFeature(1, tf.int64, 0)
            , "ord_7d": v1.FixedLenFeature(1, tf.int64, 0)
            , "pay_7d": v1.FixedLenFeature(1, tf.int64, 0)

            , "cate_id": v1.FixedLenFeature(1, tf.string, "-1")
            , "goods_id": v1.FixedLenFeature(1, tf.string, "-1")
            , "cate_level1_id": v1.FixedLenFeature(1, tf.string, "-1")
            , "cate_level2_id": v1.FixedLenFeature(1, tf.string, "-1")
            , "cate_level3_id": v1.FixedLenFeature(1, tf.string, "-1")
            , "cate_level4_id": v1.FixedLenFeature(1, tf.string, "-1")
            , "country": v1.FixedLenFeature(1, tf.string, '-1')

            # , "seq_cate_id": v1.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
            # , "seq_goods_id": v1.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
            # , "seq_cate_id": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            # , "seq_goods_id": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            , "highLevelSeqListGoods": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            , "highLevelSeqListCateId": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            , "lowerLevelSeqListGoods": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)
            , "lowerLevelSeqListCateId": v1.FixedLenFeature(20, tf.string, default_value=[""] * 20)

            , "is_clk": v1.FixedLenFeature(1, tf.int64, 0)
            , "is_pay": v1.FixedLenFeature(1, tf.int64, 0)
            , "sample_id": v1.FixedLenFeature(1, tf.string, "-1")
           }
        features = tf.io.parse_single_example(data, features=feature_describe)
        return features
    s3_file = 's3://warehouse-algo/rec/cn_rec_detail_sample_v10_ctr/ds=%s/%s' % (args.ds, args.file)
    local_file = './' + args.file
    os.system('aws s3 cp %s %s' % (s3_file, local_file))
    ds = tf.data.TFRecordDataset(local_file)
    ds2 = ds.map(parse).shuffle(args.batch_size * 10).batch(args.batch_size)
    ll = list(ds2.as_numpy_iterator())
    for it in ll:
        for name in args.names.split(','):
            print('feature_name:', name)
            print(it[name])

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
    parser.add_argument('--names', default='sample_id,highLevelSeqListGoods,lowerLevelSeqListGoods,lowerLevelSeqListCateId')
    args = parser.parse_args()
    main(args)


