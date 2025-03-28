# encoding:utf-8
import time

import tensorflow as tf
import numpy as np
import json, os

os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'
print('os.environ:', os.environ)
import pandas as pd
import sys
import pickle


class DataProcess(object):
    def __init__(self):
        self.input_feat_norm = None
    def _parse_fea(self, data):
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
           , "is_pay": tf.FixedLenFeature(1, tf.int64, 0)
           , "sample_id": tf.FixedLenFeature(1, tf.string, "-1")
       }
       features = tf.io.parse_single_example(data, features=feature_describe)

       is_clk = features.pop('is_clk')
       is_pay = features.pop('is_pay')
       if self.input_feat_norm is None and features is not None:
           print('set input_feat_norm', self.input_feat_norm)
           self.input_feat_norm = features
       print('features_data', features)
       labels = {'is_clk': tf.to_float(is_clk), 'is_pay': tf.to_float(is_pay)}

       return features, labels

    def input_fn_from_local_tfrecords(self,mode='train',batch_size=256,tfrecord_path='',
                                      num_parallel_calls=8,
                                      shuffle_factor=10, prefetch_factor=20):
        assert mode in ('train', 'eval', 'infer')

        dataset = tf.data.TFRecordDataset(tfrecord_path)
        # https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#training-with-pipe-mode-using-pipemodedataset
        dataset = dataset.map(self._parse_fea, num_parallel_calls=num_parallel_calls)
        dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)
        dataset = dataset.prefetch(buffer_size=batch_size * prefetch_factor)
        dataset = dataset.batch(batch_size)
        data_iter = dataset.make_one_shot_iterator()
        print('#' * 40, 'dataset5')
        features, labels = data_iter.get_next()
        print('raw features:', features)
        print('raw click:', labels)
        return features, labels

    def input_fn_from_csv(self):
        from ast import literal_eval
        x = pd.read_csv(filepath_or_buffer='./cn_rec_detail_sample_v1_1000.csv',
                        usecols=['ctr_7d', 'cvr_7d', 'is_clk', 'seq_goods_id', 'country', 'cate_id', 'goods_id'],
                        converters={"seq_goods_id": literal_eval},
                        dtype={'ctr_7d': np.float32, 'cvr_7d': np.float32, 'cate_id': np.str_,
                               'is_clk': np.int32, 'seq_goods_id': np.str_, 'country': np.str_,
                               'goods_id': np.str_
                               })

        print('x:', x.head(20).to_dict())
        y = x["is_clk"]
        # x['seq_goods_id'] = pd.Series(id_seq)
        # x['seq_cate_id'] = pd.Series(id_seq)
        input_fn = tf.estimator.inputs.pandas_input_fn(
            x, y=y, batch_size=20, num_epochs=1, shuffle=True, queue_capacity=1000,
            num_threads=1
        )
        return input_fn

    def input_fn_from_numpy(self):
        ctr_7d = np.array([0.1] * 1000)
        cvr_7d = np.array([0.1] * 1000)
        seq_goods_id = np.array([['1', '2', '3', '4', '5']] * 1000)
        goods_id = np.array(['4'] * 1000)
        country = np.array(['in'] * 1000)
        cate_id = np.array([2] * 1000)
        y = np.array([1] * 1000)
        x = {'ctr_7d': ctr_7d, 'cvr_7d': cvr_7d, 'seq_goods_id': seq_goods_id, 'goods_id': goods_id, 'country': country,
             'cate_id': cate_id}

        return tf.estimator.inputs.numpy_input_fn(
            x, y=y, batch_size=100, num_epochs=1, shuffle=True, queue_capacity=1000,
            num_threads=1
        )

    @tf.function
    def input_fn_from_text_dataset(self,filenames):
        dataset = tf.data.TextLineDataset(
            filenames
        )
        dataset = dataset.map(lambda x: self._parse_fea(x))
        dataset = dataset.shuffle(10).batch(20).prefetch(1)
        return dataset


def build_feature_columns():
    # cate-seq
    cateid1_fc_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket(key="cate_level1_id", hash_bucket_size=100), 8)
    cateid2_fc_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket(key="cate_level2_id", hash_bucket_size=400), 8)
    cateid3_fc_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket(key="cate_level3_id", hash_bucket_size=1000), 8)
    cateid4_fc_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket(key="cate_level4_id", hash_bucket_size=2000), 8)
    country_fc_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket(key="country", hash_bucket_size=20), 4)
    cate_cols_emb = [cateid1_fc_emb, cateid2_fc_emb, cateid3_fc_emb, cateid4_fc_emb, country_fc_emb]

    # seq and target id
    # cateid_fc = tf.feature_column.categorical_column_with_identity(key="cate_id", num_buckets=2000)
    # seq_cateid_fc = tf.feature_column.categorical_column_with_identity(key="seq_cate_id", num_buckets=2000)
    # cateid_fc_list = [cateid_fc, seq_cateid_fc]
    # goodsid_fc = tf.feature_column.categorical_column_with_hash_bucket(key="goods_id", hash_bucket_size=40000)
    # seq_goodsid_fc = tf.feature_column.categorical_column_with_hash_bucket(key="seq_goods_id", hash_bucket_size=40000)
    # goodsid_fc_list = [goodsid_fc, seq_goodsid_fc]

    #  numeric-cols
    ctr_7d = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="ctr_7d"),
                                                 boundaries=[0.0145, 0.01791, 0.01957, 0.02074, 0.02171, 0.02254, 0.02324, 0.02395, 0.02461, 0.02519, 0.02587, 0.02654, 0.02726, 0.02803, 0.02893, 0.02987, 0.03101, 0.03255, 0.0351, 0.18182])
    cvr_7d = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="cvr_7d"),
                                                 boundaries=[0.0, 0.00152, 0.00249, 0.00328, 0.00394, 0.00459, 0.00521, 0.0058, 0.00648, 0.00718, 0.00797, 0.00882, 0.00979, 0.01081, 0.0121, 0.01369, 0.01572, 0.01891, 0.0249, 42.0])
    show_7d = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="show_7d"), boundaries=[0,8279,16799,26095,38127,51407,67039,84351,103534,126813,152511,180693,210943,247295,287999,339795,412444,499499,646399,909311,2599932])
    click_7d = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="click_7d"),
                                                   boundaries=[0,174,379,621,909,1246,1625,2057,2592,3186,3835,4566,5323,6229,7439,8943,10559,12792,16163,23770,61457])
    cart_7d = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="cart_7d"), boundaries=[0,7,18,33,50,71,97,127,166,211,267,329,405,493,599,718,883,1136,1560,2314,6402])
    ord_total = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="ord_total"),
                                                    boundaries=[-1,1,7,16,26,39,56,77,104,136,177,233,310,433,605,876,1227,1939,2700,5037,82790])
    pay_total = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pay_total"),
                                                    boundaries=[-1,1,6,13,22,32,44,59,79,105,135,177,234,317,444,619,890,1305,2051,4063,64054])
    ord_7d = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="ord_7d"), boundaries=[-1,0,1,3,4,7,9,13,17,22,27,35,44,55,67,82,103,133,185,289,1979])
    pay_7d = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="pay_7d"), boundaries=[-1,0,1,2,4,6,8,11,15,19,25,32,40,49,60,73,92,119,167,260,1917])
    numric_cols = [ctr_7d, cvr_7d, show_7d, click_7d, cart_7d,ord_total, pay_total, ord_7d, pay_7d]

    return {"cate_cols_emb": cate_cols_emb,"numric_cols": numric_cols
    }

def attention_layer(seq_ids, tid_ids, id_type, shape):
    with tf.variable_scope("attention_" + id_type):
        print('raw seq_ipt tensor shape:', seq_ids.get_shape())
        print('raw tid_ipt tensor shape:', tid_ids.get_shape())
        seq_ids_hash = tf.string_to_hash_bucket_fast(seq_ids, shape[0])
        tid_ids_hash = tf.string_to_hash_bucket_fast(tid_ids, shape[0])
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                     shape=shape)
        seq_emb = tf.nn.embedding_lookup(embeddings, seq_ids_hash )
        seq_len = seq_emb.get_shape()[1]
        print('seq_emb',seq_emb.get_shape())
        tid_emb = tf.nn.embedding_lookup(embeddings, tid_ids_hash)
        print('tid_emb', tid_emb.get_shape())
        tid_emb_tile = tf.tile(tid_emb, [1, seq_len, 1])
        net = tf.concat([seq_emb, tid_emb_tile, seq_emb - tid_emb_tile, seq_emb * tid_emb_tile], axis=-1)
        for layer_id, units in enumerate([4*shape[1], 2*shape[1], 8, 1]):
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        score = tf.reshape(net, [-1, 1,  seq_len])
        # mask = tf.sequence_mask(seq_len, 30)
        # paddings = tf.zeros_like(score)
        # score_pad = tf.where(mask, score, paddings)
        score_softmax = tf.nn.softmax(score)
        output = tf.matmul(score_softmax, seq_emb)
        output_2d = tf.reduce_mean(output, axis=1)
        return output_2d

    # max_seq_len = tf.shape(seq_ids)[1] # padded_dim
    # u_emb = tf.reshape(seq_emb, shape=[-1, embedding_size])
    # a_emb = tf.reshape(tf.tile(tid_emb, [1, max_seq_len]), shape=[-1, embedding_size])
    # net = tf.concat([u_emb, u_emb - a_emb, a_emb], axis=1)
    # for units in params['attention_hidden_units']:
    #   net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    # att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)
    # att_wgt = tf.reshape(att_wgt, shape=[-1, max_seq_len, 1], name="weight")
    # wgt_emb = tf.multiply(seq_emb, att_wgt)  # shape(batch_size, max_seq_len, embedding_size)
    # #masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
    # masks = tf.expand_dims(tf.cast(seq_ids >= 0, tf.float32), axis=-1)
    # att_emb = tf.reduce_sum(tf.multiply(wgt_emb, masks), 1, name="weighted_embedding")
    # return att_emb, tid_emb

def variable_summaries(var):
  """将大量摘要附加到 Tensor 上（为了 TensorBoard 可视化）"""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class DIN(tf.estimator.Estimator):
    def __init__(self,
                 params,
                 model_dir=None,
                 optimizer='Adagrad',
                 config=None,
                 warm_start_from=None,
                 ):
        def _model_fn(features, labels, mode, params):
            print('features:', features)
            print('labels', labels)
            print('mode', mode)
            print('params', params)
            print('features seq_goods_id', features['seq_goods_id'])
            cate_cols_emb = params["feature_columns"]["cate_cols_emb"]
            cate_cols_emb_input = tf.feature_column.input_layer(features, cate_cols_emb)
            numric_cols = params["feature_columns"]["numric_cols"]
            numric_cols_emb = []
            for fc in numric_cols:
                numric_cols_emb.append(tf.feature_column.embedding_column(fc, dimension=4))
            numric_cols_emb_input =  tf.feature_column.input_layer(features, numric_cols_emb)

            seq_goodsid_input = attention_layer(seq_ids=features['seq_goods_id'],tid_ids=features['goods_id'],
                                              id_type='goods_id', shape=[40000, 32])
            seq_cateid_input = attention_layer(seq_ids=features['seq_cate_id'],tid_ids=features['cate_id'],
                                                id_type='cate_id', shape=[2000, 16])

            input_layer = [numric_cols_emb_input, cate_cols_emb_input,seq_goodsid_input, seq_cateid_input]
            # input_layer = [numric_cols_emb_input, cate_cols_emb_input]
            for ele in input_layer:
                print('blick layer shape:', ele.get_shape())
            net = tf.concat(input_layer, axis=1)
            print('input net shape:', net.get_shape())
            for units in params['hidden_units']:
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            logits = tf.layers.dense(net, units=1)
            prop = tf.sigmoid(logits, name="pred")
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'probabilities': prop,
                    # 'sample_id': features['sample_id'] # only train,eval mode not fit infer mode
                }
                export_outputs = {
                    'prediction': tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['is_clk'], logits=logits),
                                     name="loss")

            with tf.name_scope("model_din_loss"):
                variable_summaries(loss)
            accuracy = tf.metrics.accuracy(labels=labels['is_clk'],
                                               predictions=tf.to_float(tf.greater_equal(prop, 0.5)))
            auc = tf.metrics.auc(labels['is_clk'], prop)
            metrics = {'accuracy': accuracy, 'auc': auc}
            tf.summary.scalar('accuracy', accuracy[1])
            tf.summary.scalar('auc', auc[1])
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

            # Create training op.
            assert mode == tf.estimator.ModeKeys.TRAIN
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        super(DIN, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config, params=params, warm_start_from=warm_start_from)

def main(args):
    feature_columns = build_feature_columns()
    dp = DataProcess()
    estimator = DIN(
        params={
            'feature_columns': feature_columns,
            'hidden_units': args.hidden_units.split(','),
            'learning_rate': 0.001,
            'dropout_rate': 0.0001,
        },
        optimizer='Adam',
        config=tf.estimator.RunConfig(model_dir=args.model_dir, save_checkpoints_steps=args.save_checkpoints_steps)
    )
    train_input_fn = lambda: dp.input_fn_from_local_tfrecords(mode=args.mode, batch_size=args.batch_size,
                                                             tfrecord_path=args.train_path,num_parallel_calls=args.num_parallel_calls)
    eval_input_fn = lambda: dp.input_fn_from_local_tfrecords(mode=args.mode, batch_size=args.batch_size,
                                     tfrecord_path=args.eval_path,num_parallel_calls=args.num_parallel_calls)

    print('begin train', '#' * 80)
    st = time.time()
    estimator.train(input_fn=train_input_fn, max_steps=None)
    ed = time.time()
    print('end train cost:',str(ed-st), '#' * 80)

    print('begin evaluate', '#' * 80)
    st = time.time()
    results = estimator.evaluate(input_fn=eval_input_fn)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))

    ed = time.time()
    print('end evaluate cost:', str(ed - st), '#' * 80)
    # print("begin predict", '#' * 80)
    # predictions = estimator.predict(input_fn=eval_input_fn)
    # print("predictions:", predictions)
    # print("predictions:", [v['probabilities'] for v in list(predictions)])

    print('begin predict', '#' * 80)
    st = time.time()
    pred = estimator.predict(input_fn=eval_input_fn)
    pred_list = []
    for ele in pred:
        pred_list.append(ele) # ele is dict
    with open(args.pred_save_file, 'wb') as fout:
        pickle.dump(pred_list, fout)
    ed = time.time()
    print('end predict cost:', str(ed-st), '#' * 80)
    print('pred_head 100 element:', pred_list[0:100])
    print('save pred result to file:')

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train')

    def make_serving_input_receiver_fn():
        feature_columns_list = []
        for k, v in feature_columns.items():
            feature_columns_list += list(v)
        # feature_spec = tf.feature_column.make_parse_example_spec(set(feature_columns_list))
        # other_feature_spec = {
        #     "seq_goods_id": tf.FixedLenFeature(20, tf.string),
        #     "goods_id": tf.FixedLenFeature(1, tf.string),
        #     "seq_cate_id": tf.FixedLenFeature(20, tf.string),
        #     "cate_id": tf.FixedLenFeature(1, tf.string),
        # }
        # feature_spec.update(other_feature_spec)
        # print('feature_spec:', feature_spec)

        # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        print('input_feat_norm:', dp.input_feat_norm)
        json_serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(dp.input_feat_norm)
        print('json_serving_input_fn',json_serving_input_fn)
        return json_serving_input_fn
    feature_spec = {
        "ctr_7d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="ctr_7d"),
        "cvr_7d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="cvr_7d"),
        "show_7d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="show_7d"),
        "click_7d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="click_7d"),
        "cart_7d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="cart_7d"),
        "ord_total": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="ord_total"),
        "pay_total": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pay_total"),
        "ord_7d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="ord_7d"),
        "pay_7d": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="pay_7d"),
        "cate_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_id"),
        "goods_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="goods_id"),
        "cate_level1_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_level1_id"),
        "cate_level2_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_level2_id"),
        "cate_level3_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_level3_id"),
        "cate_level4_id": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_level4_id"),
        "country": tf.placeholder(dtype=tf.string, shape=[None, 1], name="country"),
        "seq_cate_id": tf.placeholder(dtype=tf.string, shape=[None, 20], name="seq_cate_id"),
        "seq_goods_id": tf.placeholder(dtype=tf.string, shape=[None, 20], name="seq_goods_id"),
    }
    print('feature_spec placeholder', feature_spec)
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    print('begin export_savemodel', '#' * 80)
    print('model_dir:', args.model_dir)
    estimator.export_savedmodel(args.model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string("mode", "train", "")
    tf.app.flags.DEFINE_integer("save_checkpoints_steps", 10000, 100)
    tf.app.flags.DEFINE_float("linear_lr", 0.005, "")
    tf.app.flags.DEFINE_float("dnn_lr", 0.01, "")
    tf.app.flags.DEFINE_float("dnn_dropout", 0.0, "")
    tf.app.flags.DEFINE_integer("batch_size", 64, "")
    tf.app.flags.DEFINE_integer("train_steps", 1000, 100)
    tf.app.flags.DEFINE_integer("epochs", 1, "")
    tf.app.flags.DEFINE_string("hidden_units", "256,128,64", "")
    tf.app.flags.DEFINE_string("checkpoint_path", "", "")
    # filename = '/home/sagemaker-user/mayfair/algo_rec/rank/exp/cn_rec_detail_sample_v1_tfr-all/ds=20241112/part-%s-827236cb-422b-4758-9f33-265565f1aad3-c000'
    filename = '/home/sagemaker-user/mayfair/algo_rec/data_process/cn_rec_detail_sample_v1_test/ds=20241113/part-%s-7736361e-69bb-4129-b5c2-da50aad1be32-c000'
    list_files = []
    for e in range(0,200):
        part = '0' * (5 - len(str(e))) + str(e)
        list_files.append(filename%(part))
    eval_list_files = list_files[0:2]
    tf.app.flags.DEFINE_list("eval_path", eval_list_files, "")
    tf.app.flags.DEFINE_list("train_path", list_files[0:2], "")
    tf.app.flags.DEFINE_integer("num_parallel_calls", 10, "")
    tf.app.flags.DEFINE_string("model_dir",'/home/sagemaker-user/mayfair/algo_rec/rank/exp/model_seq_nohead_1day_1128_sample_id', "")
    tf.app.flags.DEFINE_string("target", "ctr", "contracted")
    tf.app.flags.DEFINE_string("pred_save_file", "./pred_ret.pkl", "")
    main(FLAGS)
