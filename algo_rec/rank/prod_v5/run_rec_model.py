# encoding:utf-8
import time
import tensorflow as tf
import json, os, sys
import pickle
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'
print('os.environ:', os.environ)
from aws_auth_init import *
from build_feature_columns import build_feature_columns
from din_mask_esmm import DIN_MASK_ESMM
from feature_serv_describe import feature_describe, feature_spec_serve

def _parse_fea(data):
    print('feature_describe', feature_describe)
    features = tf.io.parse_single_example(data, features=feature_describe)

    is_clk = features.pop('is_clk')
    is_pay = features.pop('is_pay')
    print('features_data', features)
    labels = {'is_clk': tf.to_float(is_clk), 'is_pay': tf.to_float(is_pay)}

    return features, labels


def input_fn(task='ctr', batch_size=256, channel='train',
             num_parallel_calls=8,
             shuffle_factor=10, prefetch_factor=20, host_num=1, host_rank=0, site_code=None):
    from sagemaker_tensorflow import PipeModeDataset
    dataset = PipeModeDataset(channel=channel, record_format="TFRecord")
    # https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#training-with-pipe-mode-using-pipemodedataset
    dataset = dataset.shard(host_num, host_rank)
    if channel=='eval':
        dataset = dataset.take(1000)
    dataset = dataset.map(_parse_fea, num_parallel_calls=num_parallel_calls)
    if site_code is not None:
        print('only site_code:%s data use' % (str(site_code)))
        dataset = dataset.filter(lambda x, y: tf.math.equal(x['country'][0], site_code))
    dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)
    dataset = dataset.prefetch(buffer_size=batch_size * prefetch_factor)
    dataset = dataset.batch(batch_size)
    data_iter = dataset.make_one_shot_iterator()
    print('#' * 40, 'dataset5')
    features, labels = data_iter.get_next()
    print('raw features:', features)
    print('raw click:', labels)
    return features, labels


def main(args):
    host_num = len(args.hosts)
    host_rank = args.hosts.index(args.current_host)
    print('args.hosts', args.hosts, 'args.current_host', args.current_host)
    print('num_host', host_num, 'host_rank', host_rank)
    feature_columns = build_feature_columns()

    if 'seq_mask_on' in args.version:
        print('fts version:', args.version)
        feature_spec_serve.update({
            "highLevelSeqListGoods": tf.placeholder(dtype=tf.string, shape=[None, 20], name="seq_hl_goods_id"),
            "highLevelSeqListCateId": tf.placeholder(dtype=tf.string, shape=[None, 20], name="seq_hl_cate_id"),
            "lowerLevelSeqListGoods": tf.placeholder(dtype=tf.string, shape=[None, 20], name="seq_ll_goods_id"),
            "lowerLevelSeqListCateId": tf.placeholder(dtype=tf.string, shape=[None, 20], name="seq_ll_cate_id"),
            "highLevelSeqList_len": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="seq_hl_seq_len"),
            "lowerLevelSeqList_len": tf.placeholder(dtype=tf.int64, shape=[None, 1], name="seq_ll_seq_len"),
        })
        feature_describe.update({
            "highLevelSeqListGoods": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20),
            "highLevelSeqListCateId": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20),
            "lowerLevelSeqListGoods": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20),
            "lowerLevelSeqListCateId": tf.FixedLenFeature(20, tf.string, default_value=[""] * 20),
            "highLevelSeqList_len": tf.FixedLenFeature(1, tf.int64, default_value=0),
            "lowerLevelSeqList_len": tf.FixedLenFeature(1, tf.int64, default_value=0),
        })
    else:
        print(f"error version {args.version}")

    estimator = DIN_MASK_ESMM(
        params={
            'feature_columns': feature_columns,
            'hidden_units': args.hidden_units.split(','),
            'learning_rate': 0.001,
            'dropout_rate': 0.0001,
            'task': args.task,
            'version': args.version,
            'initialize': args.initialize,

        },
        optimizer='Adam',
        warm_start_from=args.warm_start_from,
        config=tf.estimator.RunConfig(model_dir=args.model_dir, save_checkpoints_steps=args.save_checkpoints_steps)
    )

    train_input_fn = lambda: input_fn(task=args.task, batch_size=args.batch_size,
                                      channel='train', num_parallel_calls=args.num_parallel_calls,
                                      host_num=host_num, host_rank=host_rank, site_code=args.site_code)
    eval_input_fn = lambda: input_fn(task=args.task, batch_size=args.batch_size,
                                     channel='eval', num_parallel_calls=args.num_parallel_calls,
                                     host_num=host_num, host_rank=host_rank, site_code=args.site_code)
    if host_rank == 0:
        time.sleep(15 * 2)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, steps=None)

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
    tf.app.flags.DEFINE_string("version", "seq_on", "seq_version:seq_on|seq_off|seq_mask_on")
    tf.app.flags.DEFINE_string("pred_local", "./predict_result.pkl", "save_pred_result_local")
    tf.app.flags.DEFINE_string("pred_s3", "s3://warehouse-algo/rec/model_pred/predict_result.pkl",
                               "save_pred_result_s3")
    tf.app.flags.DEFINE_string("warm_start_from", None, None)
    tf.app.flags.DEFINE_string("site_code", None, None)
    tf.app.flags.DEFINE_integer("num_parallel_calls", 20, 20)
    tf.app.flags.DEFINE_string("model_dir", os.environ["SM_MODEL_DIR"], "")
    tf.app.flags.DEFINE_string("initialize", 'normal', 'normal')
    print('start main', '#' * 80)
    st = time.time()
    main(FLAGS)
    ed = time.time()
    print('end main cost:%s' % (str(ed - st)), '#' * 80)
