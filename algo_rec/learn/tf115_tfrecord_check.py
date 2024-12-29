import tensorflow as tf
import argparse

# print(tf.__version__)
# a = tf.data.Dataset.range(1, 1000000)
# iter = a.map(lambda x: x + 1).shard(1,0).shuffle(10).batch(10).prefetch(10).make_one_shot_iterator()
# print(iter.get_next())


def _parse_fea(data):
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

       , "cate_id": tf.FixedLenFeature(1, tf.string, "-1")
       , "goods_id": tf.FixedLenFeature(1, tf.string, "-1")
       , "cate_level1_id": tf.FixedLenFeature(1, tf.string, "-1")
       , "cate_level2_id": tf.FixedLenFeature(1, tf.string, "-1")
       , "cate_level3_id": tf.FixedLenFeature(1, tf.string, "-1")
       , "cate_level4_id": tf.FixedLenFeature(1, tf.string, "-1")
       , "country": tf.FixedLenFeature(1, tf.string, '-1')

       # , "seq_cate_id": tf.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
       # , "seq_goods_id": tf.FixedLenSequenceFeature(20, tf.string, default_value="-1", allow_missing=True)
       , "seq_cate_id": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)
       , "seq_goods_id": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20)

       , "is_clk": tf.FixedLenFeature(1, tf.int64, 0)
       , "is_pay": tf.FixedLenFeature(1, tf.int64, 0),
       "highLevelSeqListGoods": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20),
       "highLevelSeqListCateId": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20),
       "lowerLevelSeqListGoods": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20),
       "lowerLevelSeqListCateId": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20),
       "highLevelSeqList_len": tf.FixedLenFeature(1, tf.int64, default_value=0),
       "lowerLevelSeqList_len": tf.FixedLenFeature(1, tf.int64, default_value=0),
   }
   features = tf.io.parse_single_example(data, features=feature_describe)

   is_clk = features.pop('is_clk')
   is_pay = features.pop('is_pay')
   input_feat_norm = features
   print('features_data', features)

   return features, is_clk


def main(args):

    dataset = tf.data.TFRecordDataset(args.file)
    dataset = dataset.map(_parse_fea)
    dataset = dataset.filter(lambda x, y: tf.math.equal(x['country'][0],'IN'))
    dataset = dataset.batch(args.batch_size)
    data_iter = dataset.make_one_shot_iterator()
    features, click = data_iter.get_next()
    with tf.Session() as sess:
        print(sess.run([features, click]))
    print('raw features:', features)
    print('raw click:', click)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='record',
        description='record',
        epilog='record-help')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--file', type=str, default='')
    args = parser.parse_args()
    main()