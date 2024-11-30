# encoding:utf-8

import time
import tensorflow as tf
import json, os
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'
print('os.environ:', os.environ)
from aws_auth_init import *

def _parse_fea(data):
    feature_describe = {
        "ctr_7d": tf.FixedLenFeature(1, tf.float32, 0.0)
        , "is_clk": tf.FixedLenFeature(1, tf.int64, 0)
        , "is_pay": tf.FixedLenFeature(1, tf.int64, 0)
    }
    features = tf.io.parse_single_example(data, features=feature_describe)

    is_clk = features.pop('is_clk')
    is_pay = features.pop('is_pay')
    print('features_data', features)
    labels = {'is_clk': tf.to_float(is_clk), 'is_pay': tf.to_float(is_pay)}

    return features, labels


def input_fn(mode='train', batch_size=256, channel='train',
             num_parallel_calls=8,
             shuffle_factor=10, prefetch_factor=20, host_num=1, host_rank=0):
    assert mode in ('train', 'eval', 'infer')

    from sagemaker_tensorflow import PipeModeDataset
    dataset = PipeModeDataset(channel=channel, record_format="TFRecord")
    # https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#training-with-pipe-mode-using-pipemodedataset
    dataset = dataset.shard(host_num, host_rank)
    dataset = dataset.map(_parse_fea, num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)
    dataset = dataset.prefetch(buffer_size=batch_size * prefetch_factor)
    dataset = dataset.batch(batch_size)
    data_iter = dataset.make_one_shot_iterator()
    print('#' * 40, 'dataset5')
    features, labels = data_iter.get_next()
    print('raw features:', features)
    print('raw click:', labels)
    return features, labels


def build_feature_columns():
    #  numeric-cols
    ctr_7d = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="ctr_7d"),
                                                 boundaries=[0.0145, 0.01791, 0.01957, 0.02074, 0.02171, 0.02254, 0.02324, 0.02395, 0.02461, 0.02519, 0.02587, 0.02654, 0.02726, 0.02803, 0.02893, 0.02987, 0.03101, 0.03255, 0.0351, 0.18182])
    numric_cols = [ctr_7d]

    return {"numric_cols": numric_cols }


class DIN(tf.estimator.Estimator):
    def __init__(self,
                 params,
                 model_dir=None,
                 config=None,
                 warm_start_from=None,
                 ):
        def _model_fn(features, labels, mode, params):
            print('features:', features)
            print('labels', labels)
            print('mode', mode)
            print('params', params)
            numric_cols = params["feature_columns"]["numric_cols"]
            numric_cols_emb = []
            for fc in numric_cols:
                numric_cols_emb.append(tf.feature_column.embedding_column(fc, dimension=4))
            numric_cols_emb_input =  tf.feature_column.input_layer(features, numric_cols_emb)
            input_layer = [numric_cols_emb_input,numric_cols_emb_input]
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

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss)

            # Create training op.
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        super(DIN, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config, params=params, warm_start_from=warm_start_from)

def main(args):
    host_num = len(args.hosts)
    host_rank = args.hosts.index(args.current_host)
    print('args.hosts', args.hosts, 'args.current_host', args.current_host)
    print('num_host', host_num, 'host_rank', host_rank)
    feature_columns = build_feature_columns()
    estimator = DIN(
        params={
            'feature_columns': feature_columns,
            'hidden_units': args.hidden_units.split(','),
            'learning_rate': 0.001,
            'dropout_rate': 0.0001,
        },
        config=tf.estimator.RunConfig(model_dir=args.model_dir, save_checkpoints_steps=args.save_checkpoints_steps)
    )
    train_input_fn = lambda: input_fn(mode=args.mode, batch_size=args.batch_size,
                                         channel='train', num_parallel_calls=args.num_parallel_calls,
                                      host_num=host_num, host_rank=host_rank)

    print('begin train', '#' * 80)
    st = time.time()
    estimator.train(input_fn=train_input_fn, max_steps=None)
    ed = time.time()
    print('end train cost:',str(ed-st), '#' * 80)

    if host_rank == 0:
        feature_spec = {
            "ctr_7d": tf.placeholder(dtype=tf.float32, shape=[None, 1], name="ctr_7d"),
        }
        print('feature_spec placeholder', feature_spec)
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        print('begin export_savemodel', '#' * 80)
        print('model_dir:', args.model_dir)
        estimator.export_savedmodel(args.model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_list("hosts", json.loads(os.environ.get("SM_HOSTS")), [])
    tf.app.flags.DEFINE_string("current_host", os.environ.get("SM_CURRENT_HOST"), [])
    tf.app.flags.DEFINE_string("mode", "train", "")
    tf.app.flags.DEFINE_integer("save_checkpoints_steps", 100, 100)
    tf.app.flags.DEFINE_integer("batch_size", 1024, 1024)
    tf.app.flags.DEFINE_string("hidden_units", "256,128,64", "")
    tf.app.flags.DEFINE_integer("num_parallel_calls", 15, 10)
    tf.app.flags.DEFINE_string("model_dir",os.environ["SM_MODEL_DIR"], "")
    print('start main', '#' * 80)
    st = time.time()
    main(FLAGS)
    ed = time.time()
    print('end main cost:%s'%(str(ed-st)), '#' * 80)
