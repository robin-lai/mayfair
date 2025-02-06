# encoding:utf-8
import tensorflow as tf
from attention import attention_layer_mask
class DIN_MASK_CTR(tf.estimator.Estimator):
    def __init__(self,
                 params,
                 model_dir=None,
                 optimizer='Adagrad',
                 config=None,
                 warm_start_from=None,
                 ):

        def _model_fn_ctr(features, labels, mode, params):
            print('current task:', params['task'])
            print('features:', features)
            print('labels', labels)
            print('mode', mode)
            print('params', params)
            cate_cols_emb = params["feature_columns"]["cate_cols_emb"]
            cate_cols_emb_input = tf.feature_column.input_layer(features, cate_cols_emb)
            numric_cols_emb = params["feature_columns"]["numric_cols_emb"]
            numric_cols_emb_input = tf.feature_column.input_layer(features, numric_cols_emb)
            input_layer = [numric_cols_emb_input, cate_cols_emb_input]


            if 'seq_mask_on' in params['version']:
                seq_high_on_goodsid_input = attention_layer_mask(seq_ids=features['highLevelSeqListGoods'],
                                                                 tid_ids=features['goods_id'],
                                                                 id_type='seq_on_high_goods_id', shape=[40000, 32],
                                                                 att_type='dot'
                                                                 , seq_len_actual=features['highLevelSeqList_len'],
                                                                 max_len=20, initialize=params['initialize'])
                seq_high_on_cateid_input = attention_layer_mask(seq_ids=features['highLevelSeqListCateId'],
                                                                tid_ids=features['cate_id'],
                                                                id_type='seq_on_high_cate_id', shape=[2000, 16],
                                                                att_type='dot'
                                                                , seq_len_actual=features['highLevelSeqList_len'],
                                                                max_len=20, initialize=params['initialize'])

                seq_low_on_goodsid_input = attention_layer_mask(seq_ids=features['lowerLevelSeqListGoods'],
                                                                tid_ids=features['goods_id'],
                                                                id_type='seq_on_low_goods_id', shape=[40000, 32],
                                                                att_type='dot'
                                                                , seq_len_actual=features['lowerLevelSeqList_len'],
                                                                max_len=20, initialize=params['initialize'])
                seq_low_on_cateid_input = attention_layer_mask(seq_ids=features['lowerLevelSeqListCateId'],
                                                               tid_ids=features['cate_id'],
                                                               id_type='seq_on_low_cate_id', shape=[2000, 16],
                                                               att_type='dot'
                                                               , seq_len_actual=features['lowerLevelSeqList_len'],
                                                               max_len=20, initialize=params['initialize'])
                input_layer.extend([seq_high_on_cateid_input, seq_high_on_goodsid_input, seq_low_on_cateid_input,
                                    seq_low_on_goodsid_input])

            # input_layer = [numric_cols_emb_input, cate_cols_emb_input]
            for ele in input_layer:
                print('blick layer shape:', ele.get_shape())
            net = tf.concat(input_layer, axis=1)
            print('input net shape:', net.get_shape())
            with tf.variable_scope('ctr_model'):
                for units in params['hidden_units']:
                    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
                logits = tf.layers.dense(net, units=1)
                prop = tf.sigmoid(logits, name="pred")
            if mode == tf.estimator.ModeKeys.PREDICT:
                print('modekeys predict:', mode)
                predictions = {
                    'ctr': prop,
                    'cvr': prop
                }
                export_outputs = {
                    'prediction': tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['is_clk'], logits=logits),
                                  name="loss")

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss)

            # Create training op.
            assert mode == tf.estimator.ModeKeys.TRAIN
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        # elif params['task'] == 'ctr':
        if warm_start_from is None:
            super(DIN_MASK_CTR, self).__init__(
                model_fn=_model_fn_ctr, model_dir=model_dir, config=config, params=params)
        else:
            super(DIN_MASK_CTR, self).__init__(
                model_fn=_model_fn_ctr, model_dir=model_dir, config=config, params=params,
                warm_start_from=warm_start_from)
