# encoding:utf-8
import tensorflow as tf
# import tensorflow.compat.v1 as tf1

# from sagemaker_tensorflow import PipeModeDataset
import json, time, sys, traceback
import os
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'
print('os.environ:',os.environ)
from algo_rec.rank.aws_auth_init import *


def input_fn(mode, channel=None, feature_description=None, label=None, batch_size=256, num_epochs=1,
             num_parallel_calls=8,
             shuffle_factor=10, prefetch_factor=20,
             fn_mode='', num_host=1, host_rank=0):
    assert mode in ('ctr', 'cvr', 'mtl')

    def _parse_fea(data):
        feature_describe = {"ctr_7d": tf.io.FixedLenFeature([1], tf.float32, 0.0)
            , "cate_id": tf.io.FixedLenFeature([1], tf.int64, 0)
            , "is_clk": tf.io.FixedLenFeature([1], tf.int64, 0)
            , "is_pay": tf.io.FixedLenFeature([1], tf.int64, 0)
                            }

        try:
            features = tf.parse_single_example(data, features=feature_describe)
        except AttributeError:
            features = tf.io.parse_single_example(data, features=feature_describe)

        is_clk = features.pop('is_clk')
        is_pay = features.pop('is_pay')
        input_feat_norm = features
        print('features_data', features)

        return features, is_clk
    print('Begin_input_fn channel', channel, '#' * 80)

    dataset = PipeModeDataset(channel=channel, record_format="TFRecord")
    # https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#training-with-pipe-mode-using-pipemodedataset
    if fn_mode == 'MultiWorkerShard':
        dataset = dataset.shard(num_host, host_rank)
    dataset = dataset.map(_parse_fea, num_parallel_calls=num_parallel_calls)

    dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)
    dataset = dataset.batch(batch_size)

    if channel == 'eval':
        print('Begin read eval data sample 100000', '#' * 80)
        dataset = dataset.take(100000)

    if prefetch_factor > 0:
        dataset = dataset.prefetch(buffer_size=prefetch_factor)
    try:
        data_iter = dataset.make_one_shot_iterator()
    except AttributeError:
        data_iter = tf.compat.v1.data.make_one_shot_iterator(dataset)

    features, click = data_iter.get_next()
    return features, click


def build_feature_columns():
    wide_fc = []
    deep_fc = []

    cateid_fc = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(key="cate_id", num_buckets=2000),
        32, combiner='mean')
    ctr_7d = tf.feature_column.numeric_column(key="ctr_7d")
    wide_fc.append(ctr_7d)
    deep_fc.append(cateid_fc)
    print("wide-fc", wide_fc)
    print("deep-fc", deep_fc)
    return wide_fc, deep_fc


def model_fn(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_feature_columns()
    hidden_units = [256, 128, 64]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    # run_config = tf.estimator.RunConfig().replace(
    #    session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns)
        # config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units)
        # config=run_config)
    elif model_type == 'wdl':
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units)
        # config=run_config)


def main(args):
    if args.mode != "train":
        print("[WARN]Unknown Mode", args.mode)
        return
    try:
        estimator = model_fn(model_dir=args.checkpoint, model_type='wdl')

        num_host = len(args.hosts)
        host_rank = args.hosts.index(args.current_host)
        print('num_host', num_host, 'host_rank', host_rank)
        print('args.hosts', args.hosts, 'args.current_host', args.current_host)
        if host_rank == 0:
            print('sleep 15*2')
            time.sleep(15 * 2)
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(mode=args.target, channel="train", batch_size=args.batch_size,
                                         fn_mode='MultiWorkerShard', num_host=num_host, host_rank=host_rank)
        ,max_steps=None)

        def make_serving_input_receiver_fn():
            wide_columns, deep_columns = build_feature_columns()
            feature_columns_new = set(wide_columns + deep_columns)
            ##print(feature_columns_new)
            feature_spec = tf.feature_column.make_parse_example_spec(feature_columns_new)
            # print(feature_spec)
            serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
            return serving_input_receiver_fn
        exporter = tf.estimator.FinalExporter(
            name="estimate",
            serving_input_receiver_fn=make_serving_input_receiver_fn()
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(mode=args.target, channel="eval", batch_size=args.batch_size),
            steps=None,
            # exporters=[exporter]
        )
        print('begin train','#'*80)
        estimator.train(input_fn=lambda: input_fn(mode=args.target, channel="train", batch_size=args.batch_size,
                                         fn_mode='MultiWorkerShard', num_host=num_host, host_rank=host_rank)
        ,max_steps=None)
        print('end train','#'*80)

        print('begin evaluate','#'*80)
        results = estimator.evaluate(input_fn=lambda: input_fn(mode=args.target, channel="eval", batch_size=args.batch_size,
                                                               fn_mode='MultiWorkerShard', num_host=num_host, host_rank=host_rank))
        print('end evaluate','#'*80)
        for key in sorted(results):
            print('%s: %s' % (key, results[key]))
        if host_rank == 0:
            print('begin export_savemodel','#'*80)
            estimator.export_savedmodel(os.environ["SM_MODEL_DIR"], make_serving_input_receiver_fn())

        time.sleep(15 * 2)
        sys.exit(0)
    except Exception as e:
        print('run train_entrance_dnn fail')
        print(Exception, e)
        traceback.print_exc()


if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_list("hosts", json.loads(os.environ.get("SM_HOSTS")), "")
    tf.app.flags.DEFINE_string("current_host", os.environ.get("SM_CURRENT_HOST"), "")
    tf.app.flags.DEFINE_integer("num_cpus", os.environ.get("SM_NUM_CPUS"), "")

    tf.app.flags.DEFINE_float("linear_lr", 0.005, "")
    tf.app.flags.DEFINE_float("dnn_lr", 0.01, "")
    tf.app.flags.DEFINE_float("dnn_dropout", 0.0, "")
    tf.app.flags.DEFINE_integer("batch_size", 512, "")
    tf.app.flags.DEFINE_integer("epochs", 1, "")
    tf.app.flags.DEFINE_string("dnn_hidden_units", "256,128,64", "")
    tf.app.flags.DEFINE_string("checkpoint", "", "")
    tf.app.flags.DEFINE_string("mode", "train", "")
    tf.app.flags.DEFINE_string("model_dir", "", "contracted")
    tf.app.flags.DEFINE_string("target", "ctr", "contracted")

    main(FLAGS)