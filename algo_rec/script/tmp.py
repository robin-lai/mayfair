%%writefile
process_data_v0.py

# encoding:utf-8

import tensorflow as tf
from tensorflow.python.ops import string_ops
from sagemaker_tensorflow import PipeModeDataset

from tensorflow.python.ops.parsing_ops import FixedLenFeature
from deepctr.estimator.inputs import input_fn_tfrecord


class DataProcessor(object):
    def __init__(self):
        self.input_feat_norm = None
        self.wide_fc = []
        self.deep_fc = []

        self.single_string_features = ['uuid', 'country', 'query', 'title', ]
        self.multi_string_features = ['query_seg', 'title_seg', ]
        # self.sequence_features = ['high_level_seq', 'low_level_seq', 'attrs', 'hist_long', 'hist_cart']
        self.sequence_features = {'high_level_seq': 20, 'low_level_seq': 20, 'attrs': 20, 'hist_long': 40,
                                  'hist_cart': 40}
        self.single_int_feautres = ['goods_id', 'cid1', 'cid2', 'cid3', 'cid4', 'retarget', 'price', 'rebuy']
        self.single_cross_string_features = ['query_goods_id', 'query_cid1', 'query_cid2', 'query_cid3', 'query_cid4',
                                             'country_goods_id', 'country_cid1', 'country_cid2', 'country_cid3',
                                             'country_cid4']
        self.multi_cross_string_features = ['query_seg_goods_id', 'query_seg_cid1', 'query_seg_cid2',
                                            'query_seg_cid3', 'query_seg_cid4', ]

        ids_map = dict()
        emb_map = dict()
        for key in self.single_string_features + self.multi_string_features + list(self.sequence_features.keys()):
            ids = ids_map[key] = tf.feature_column.categorical_column_with_hash_bucket(
                key=key, hash_bucket_size=1000000, dtype=tf.string)
            emb = emb_map[key] = tf.feature_column.embedding_column(ids, 32, combiner='sum')
            print(key, ids, emb)
            self.wide_fc.append(ids)
            self.deep_fc.append(emb)

        #         for key in self.single_int_feautres:
        #             bucket_size = 3000000 if key == 'goods_id' else 100000
        #             ids = ids_map[key] = tf.feature_column.categorical_column_with_identity(key, bucket_size)
        #             emb = emb_map[key] = tf.feature_column.embedding_column(ids, 32)
        #             print(key, ids, emb)
        #             self.wide_fc.append(ids)
        #             self.deep_fc.append(emb)
        # The former int feature processor has overflow risk, so the better solution is hash them.
        for key in self.single_int_feautres:
            ids = ids_map[key] = tf.feature_column.categorical_column_with_hash_bucket(
                key, hash_bucket_size=1000000, dtype=tf.int32)
            emb = emb_map[key] = tf.feature_column.embedding_column(ids, 32)
            print(key, ids, emb)
            self.wide_fc.append(ids)
            self.deep_fc.append(emb)

        # Enable cross feature
        for key in self.single_cross_string_features + self.multi_cross_string_features:
            ids = ids_map[key] = tf.feature_column.categorical_column_with_hash_bucket(
                key=key, hash_bucket_size=1000000, dtype=tf.string)
            self.wide_fc.append(ids)
        print(self.wide_fc)
        print(self.deep_fc)

    def _parse_fea(self, data):
        feature_description = {k: FixedLenFeature(dtype=tf.string, shape=1) for k in self.single_string_features}
        feature_description.update({k: FixedLenFeature(dtype=tf.int64, shape=1) for k in self.single_int_feautres})
        feature_description.update({k: FixedLenFeature(dtype=tf.string, shape=20) for k in self.multi_string_features})
        feature_description.update(
            {k: FixedLenFeature(dtype=tf.string, shape=v) for k, v in self.sequence_features.items()})
        feature_description.update(
            {k: FixedLenFeature(dtype=tf.float32, shape=1) for k in ['click_label', 'pay_label']})

        try:
            features = tf.parse_single_example(data, features=feature_description)
        except AttributeError:
            features = tf.io.parse_single_example(data, features=feature_description)

        click_labels = features.pop('click_label')
        pay_labels = features.pop('pay_label')
        self.input_feat_norm = features
        print(features)

        # Auto cross by concat string and int.
        int_str_tensors = [[k, tf.as_string(features[k])] for k in ['goods_id', 'cid1', 'cid2', 'cid3', 'cid4']]
        int_strs_tensors = [[k, tf.tile(v, [20])] for k, v in int_str_tensors]
        for a in ['query', 'query_seg', 'country']:
            for b, bt in (int_str_tensors if 'seg' not in a else int_strs_tensors):
                features[a + '_' + b] = tf.strings.join([features[a], ',,', bt])
        return features, click_labels, pay_labels

    def input_fn(self, mode, channel=None, feature_description=None, label=None, batch_size=256, num_epochs=1,
                 num_parallel_calls=8,
                 shuffle_factor=10, prefetch_factor=20,
                 fn_mode='', num_host=1, host_rank=0):
        assert mode in ('ctr', 'cvr', 'mtl')

        dataset = PipeModeDataset(channel=channel, record_format="TFRecord")
        if fn_mode == 'MultiWorkerShard':
            dataset = dataset.shard(num_host, host_rank)
        dataset = dataset.map(self._parse_fea, num_parallel_calls=num_parallel_calls)
        if mode == 'cvr':
            dataset = dataset.filter(lambda feature, click, pay: click[0] > 0.5)
        dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)
        dataset = dataset.batch(batch_size)  # .take(100)
        if channel == 'eval':
            print('Begin_eval_now', '#' * 80)
            dataset = dataset.take(10)

        if prefetch_factor > 0:
            dataset = dataset.prefetch(buffer_size=prefetch_factor)
        try:
            data_iter = dataset.make_one_shot_iterator()
        except AttributeError:
            data_iter = tf.compat.v1.data.make_one_shot_iterator(dataset)

        features, click, pay = data_iter.get_next()
        if mode == 'ctr':
            return features, click
        if mode == 'cvr':
            return features, pay
        if mode == 'mtl':
            return features, {'ctr': click, 'cvr': pay}
        return features, click, pay
