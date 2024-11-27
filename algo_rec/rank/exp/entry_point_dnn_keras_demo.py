# encoding:utf-8
import os
# TensorFlow is the only backend that supports string inputs.
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf # use tf=2.14

from tensorflow.keras import layers
import tensorflow.keras as keras
from sagemaker_tensorflow import PipeModeDataset
import json
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
    # dataset = tf.data.TFRecordDataset("s3://warehouse-algo/rec/cn_rec_detail_sample_v0_tfr-all/ds=20241101")
    # https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#training-with-pipe-mode-using-pipemodedataset
    # if fn_mode == 'MultiWorkerShard':
    #     dataset = dataset.shard(num_host, host_rank)
    dataset = dataset.map(_parse_fea).take(10000)

    # dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)
    dataset = dataset.batch(batch_size)

    # if channel == 'eval':
    #     print('Begin_eval_now', '#' * 80)
    #     dataset = dataset.take(10)

    # if prefetch_factor > 0:
    #     dataset = dataset.prefetch(buffer_size=prefetch_factor)
    # try:
    #     data_iter = dataset.make_one_shot_iterator()
    # except AttributeError:
    #     data_iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
    #
    # features, click = data_iter.get_next()
    return dataset



def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = layers.Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = layers.StringLookup if is_string else layers.IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="int")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


def main(args):
    num_host = len(args.hosts)
    host_rank = args.hosts.index(args.current_host)

    train_ds = input_fn(mode=args.target, channel="train", batch_size=args.batch_size,
                                fn_mode='MultiWorkerShard', num_host=num_host, host_rank=host_rank)
    val_ds = input_fn(mode=args.target, channel="train", batch_size=args.batch_size,
                                fn_mode='MultiWorkerShard', num_host=num_host, host_rank=host_rank)
    # Categorical features encoded as integers
    ctr_7d = keras.Input(shape=(1,), name="ctr_7d", dtype="float32")
    cate_id = keras.Input(shape=(1,), name="cate_id", dtype="int64")

    all_inputs = [
        ctr_7d,
        # cate_id,

    ]

    # Integer categorical features
    cate_id_encoded = encode_categorical_feature(cate_id, "cate_id", train_ds, False)
    cate_id_emb = layers.Embedding(cate_id_encoded.get_vocabulary())
    # Numerical features
    ctr_7d_encoded = encode_numerical_feature(ctr_7d, "ctr_7d", train_ds)


    all_features = layers.concatenate(
        [
            # cate_id_encoded,
            ctr_7d_encoded,

        ]
    )
    x = layers.Dense(32, activation="relu")(all_features)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(all_inputs, output)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=1, validation_data=val_ds)



if __name__ == "__main__":
    FLAGS = tf.compat.v1.flags.FLAGS
    tf.compat.v1.flags.DEFINE_list("hosts", json.loads(os.environ.get("SM_HOSTS")), "")
    tf.compat.v1.flags.DEFINE_string("current_host", os.environ.get("SM_CURRENT_HOST"), "")
    tf.compat.v1.flags.DEFINE_integer("num_cpus", os.environ.get("SM_NUM_CPUS"), "")

    tf.compat.v1.flags.DEFINE_float("linear_lr", 0.005, "")
    tf.compat.v1.flags.DEFINE_float("dnn_lr", 0.01, "")
    tf.compat.v1.flags.DEFINE_float("dnn_dropout", 0.0, "")
    tf.compat.v1.flags.DEFINE_integer("batch_size", 512, "")
    tf.compat.v1.flags.DEFINE_integer("epochs", 1, "")
    tf.compat.v1.flags.DEFINE_string("dnn_hidden_units", "256,128,64", "")
    tf.compat.v1.flags.DEFINE_string("checkpoint", "", "")
    tf.compat.v1.flags.DEFINE_string("mode", "train", "")
    tf.compat.v1.flags.DEFINE_string("model_dir", "", "contracted")
    tf.compat.v1.flags.DEFINE_string("target", "ctr", "contracted")

    main(FLAGS)