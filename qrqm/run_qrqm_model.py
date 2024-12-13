# encoding:utf-8

import time
import tensorflow as tf
import json, os,sys
import argparse
import pickle
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'
print('os.environ:', os.environ)
from aws_auth_init import *

feature_spec_serve = {
                "uuid": tf.placeholder(dtype=tf.string, shape=[None, 1], name="cate_level1_id"),
                "pgid": tf.placeholder(dtype=tf.string, shape=[None, 1], name="pgid"),
            }

feature_describe = {
        "uuid": tf.FixedLenFeature(1, tf.string, "-1")
        , "pgid": tf.FixedLenFeature(1, tf.string, "-1")
        , "is_clk": tf.FixedLenFeature(1, tf.int64, 0)

    }

def _parse_fea(data):

    print('feature_describe', feature_describe)
    features = tf.io.parse_single_example(data, features=feature_describe)

    is_clk = features.pop('is_clk')
    print('features_data', features)
    labels = {'is_clk': tf.to_float(is_clk)}

    return features, labels


def input_fn(task='ctr', batch_size=256, channel='train',
             num_parallel_calls=8,
             shuffle_factor=10, prefetch_factor=20, host_num=1, host_rank=0):

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
    if task == 'ctr':
        return features, labels['is_clk']
    elif task == 'cvr':
        return features, labels['is_pay']
    elif task == 'mtl':
        return features, labels
    else:
        print('unknown task:', task)


class DNN(tf.estimator.Estimator):
    def __init__(self,
                 params,
                 model_dir=None,
                 optimizer='Adagrad',
                 config=None,
                 warm_start_from=None,
                 ):
        def _model_fn(features, labels, mode, params):
            print('current task:', params['task'])
            print('features:', features)
            print('labels', labels)
            print('mode', mode)
            print('params', params)
            uuid = features['uuid']
            pgid = features['pgid']
            # with tf.variable_scope("qrqm_uuid_emb"):
            uuid_shape = [400000, 50]
            pgid_shape = [40000, 50]
            uuid_hash = tf.string_to_hash_bucket_fast(uuid, uuid_shape[0])
            pgid_hash = tf.string_to_hash_bucket_fast(pgid, pgid_shape[0])
            with tf.variable_scope("qrqm_emb_uuid", reuse=tf.AUTO_REUSE) as name:
                embeddings = tf.get_variable(name="qrqm_emb_uuid_v", dtype=tf.float32,
                                             shape=uuid_shape, trainable=True,
                                             initializer=tf.glorot_uniform_initializer())
                uuid_emb = tf.nn.embedding_lookup(embeddings, uuid_hash)

            with tf.variable_scope("qrqm_emb_pgid", reuse=tf.AUTO_REUSE) as name:
                embeddings = tf.get_variable(name="qrqm_emb_pgid_v", dtype=tf.float32,
                                             shape=uuid_shape, trainable=True,
                                             initializer=tf.glorot_uniform_initializer())
                pgid_emb = tf.nn.embedding_lookup(embeddings, pgid_hash)

            uuid_emb = tf.reshape(uuid_emb, shape=[-1, 50])
            pgid_emb = tf.reshape(pgid_emb, shape=[-1, 50])
            input_layer = [uuid_emb, pgid_emb]
            for ele in input_layer:
                print('block layer shape:', ele.get_shape())
            net = tf.concat(input_layer, axis=1)
            print('input net shape:', net.get_shape())
            for units in params['hidden_units']:
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            logits = tf.layers.dense(net, units=1)
            prop = tf.sigmoid(logits, name="pred")
            if mode == tf.estimator.ModeKeys.PREDICT:
                print('modekeys predict:', mode)
                predictions = {
                    'probabilities': prop,
                }
                export_outputs = {
                    'prediction': tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits),
                                     name="loss")
            accuracy = tf.metrics.accuracy(labels=labels,
                                               predictions=tf.to_float(tf.greater_equal(prop, 0.5)))
            auc = tf.metrics.auc(labels, prop)
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


        if warm_start_from is None:
            super(DNN, self).__init__(
                model_fn=_model_fn, model_dir=model_dir, config=config, params=params)
        else:
            super(DNN, self).__init__(
                model_fn=_model_fn, model_dir=model_dir, config=config, params=params, warm_start_from=warm_start_from)


def main(args):
    host_num = len(args.hosts)
    host_rank = args.hosts.index(args.current_host)
    print('args.hosts', args.hosts, 'args.current_host', args.current_host)
    print('num_host', host_num, 'host_rank', host_rank)
    feature_columns = {}


    estimator = DNN(
        params={
            'feature_columns': feature_columns,
            'hidden_units': args.hidden_units.split(','),
            'learning_rate': 0.001,
            'dropout_rate': 0.0001,
            'task': args.task,
            'version': args.version

        },
        optimizer='Adam',
        warm_start_from=args.warm_start_from,
        config=tf.estimator.RunConfig(model_dir=args.model_dir, save_checkpoints_steps=args.save_checkpoints_steps)
    )

    train_input_fn = lambda: input_fn(task=args.task, batch_size=args.batch_size,
                                      channel='train', num_parallel_calls=args.num_parallel_calls,
                                      host_num=host_num, host_rank=host_rank)
    eval_input_fn = lambda: input_fn(task=args.task, batch_size=args.batch_size,
                                     channel='eval', num_parallel_calls=args.num_parallel_calls,
                                     host_num=host_num, host_rank=host_rank)
    if host_rank == 0:
        time.sleep(15 * 2)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn
        , throttle_secs=300, steps=100)

    if args.mode == 'infer':
        print('begin predict', '#' * 80)
        st = time.time()
        pred = estimator.predict(input_fn=eval_input_fn)
        pred_list = []
        for ele in pred:
            pred_list.append(ele)  # ele is dict
        with open(args.pred_local, 'wb') as fout:
            pickle.dump(pred_list, fout)
        ed = time.time()
        print('upload %s -> %s' % (args.pred_local, args.pred_s3))
        os.system('aws s3 cp %s %s' % (args.pred_local, args.pred_s3))
        print('end predict cost:', str(ed - st), '#' * 80)
        print('pred_head 100 element:', pred_list[0:100])

    if args.mode == 'train':
        print("before train and evaluate")
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        print("after train and evaluate")

        if host_rank == 0:
            print('feature_spec placeholder', feature_spec_serve)
            serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec_serve)
            print('begin export_savemodel', '#' * 80)
            print('model_dir:', args.model_dir)
            # TODO why call model_fn with infer mode
            estimator.export_savedmodel(args.model_dir, serving_input_receiver_fn)
            sys.exit(0)
    time.sleep(15 * 2)
    sys.exit(0)


if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_list("hosts", json.loads(os.environ.get("SM_HOSTS")), "")
    tf.app.flags.DEFINE_string("current_host", os.environ.get("SM_CURRENT_HOST"), "")
    tf.app.flags.DEFINE_string("mode", "train", "")
    tf.app.flags.DEFINE_integer("save_checkpoints_steps", 10000, 100)
    tf.app.flags.DEFINE_integer("batch_size", 1024, "")
    tf.app.flags.DEFINE_string("hidden_units", "256,128,64", "")
    tf.app.flags.DEFINE_string("task", "ctr", "ctr")
    tf.app.flags.DEFINE_string("version", "seq_on", "seq_version:seq_on|seq_off")
    tf.app.flags.DEFINE_string("pred_local", "./predict_result.pkl", "save_pred_result_local")
    tf.app.flags.DEFINE_string("pred_s3", "s3://warehouse-algo/rec/model_pred/predict_result.pkl", "save_pred_result_s3")
    tf.app.flags.DEFINE_string("warm_start_from", None, None)
    tf.app.flags.DEFINE_integer("num_parallel_calls", 20, 20)
    tf.app.flags.DEFINE_string("model_dir",os.environ["SM_MODEL_DIR"], "")
    print('start main', '#' * 80)
    st = time.time()
    main(FLAGS)
    ed = time.time()
    print('end main cost:%s'%(str(ed-st)), '#' * 80)
