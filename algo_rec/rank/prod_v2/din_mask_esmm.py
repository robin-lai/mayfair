# encoding:utf-8
import tensorflow as tf
from attention import attention_layer_mask_v2
class DIN_MASK_ESMM(tf.estimator.Estimator):
    def __init__(self,
                 params,
                 model_dir=None,
                 optimizer='Adagrad',
                 config=None,
                 warm_start_from=None,
                 ):

        def _model_fn_esmm(features, labels, mode, params):
            print('current task:', params['task'])
            print('features:', features)
            print('labels', labels)
            print('mode', mode)
            print('params', params)
            cate_cols_emb = params["feature_columns"]["cate_cols_emb"]
            cate_cols_emb_input = tf.feature_column.input_layer(features, cate_cols_emb)
            numric_cols_emb = params["feature_columns"]["numric_cols_emb"]
            numric_cols_emb_input = tf.feature_column.input_layer(features, numric_cols_emb)
            cate_cols_share_emb = params["feature_columns"]["cate_cols_share_emb"]
            cate_cols_share_emb_input = tf.feature_column.input_layer(features, cate_cols_share_emb)
            input_layer = [numric_cols_emb_input, cate_cols_emb_input, cate_cols_share_emb_input]


            if 'seq_mask_on' in params['version']:
                seq_on_goodsid_input = attention_layer_mask_v2(seq_ids_high=features['highLevelSeqListGoods'],
                                                               seq_ids_low=features['lowerLevelSeqListGoods'],
                                                               seq_len_actual_high=features['highLevelSeqList_len'],
                                                               seq_len_actual_low=features['lowerLevelSeqList_len'],
                                                               tid_ids=features['goods_id'],
                                                               id_type='seq_on_goods_id', shape=[40000, 32],
                                                               att_type='dot',
                                                               max_len=20, initialize=params['initialize'])
                seq_on_cateid_input = attention_layer_mask_v2(seq_ids_high=features['highLevelSeqListCateId'],
                                                              seq_ids_low=features['lowerLevelSeqListCateId'],
                                                              seq_len_actual_high=features['highLevelSeqList_len'],
                                                              seq_len_actual_low=features['lowerLevelSeqList_len'],
                                                              tid_ids=features['cate_id'],
                                                              id_type='seq_on_cate_id', shape=[2000, 16],
                                                              att_type='dot',
                                                              max_len=20, initialize=params['initialize'])

                input_layer.extend(seq_on_cateid_input)
                input_layer.extend(seq_on_goodsid_input)

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
            # cvr
            with tf.variable_scope('cvr_model'):
                for units in params['hidden_units']:
                    net_cvr = tf.layers.dense(net, units=units, activation=tf.nn.relu)
                logits_cvr = tf.layers.dense(net_cvr, units=1)
                prop_cvr = tf.sigmoid(logits_cvr, name="pred_cvr")
            ctcvr = tf.multiply(prop, prop_cvr, name="CTCVR")
            if mode == tf.estimator.ModeKeys.PREDICT:
                print('modekeys predict:', mode)
                predictions = {
                    'ctr': prop,
                    'cvr': prop_cvr,
                    'ctcvr': ctcvr
                }
                export_outputs = {
                    'prediction': tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['is_clk'], logits=logits),
                                  name="loss")
            loss_cvr = tf.reduce_mean(tf.keras.backend.binary_crossentropy(labels['is_pay'], ctcvr), name="loss")
            loss = tf.add(loss, loss_cvr, name="ctcvr_loss")

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss)

            # Create training op.
            assert mode == tf.estimator.ModeKeys.TRAIN
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        # if params['task'] == 'mtl':
        if warm_start_from is None:
            super(DIN_MASK_ESMM, self).__init__(
                model_fn=_model_fn_esmm, model_dir=model_dir, config=config, params=params)
        else:
            super(DIN_MASK_ESMM, self).__init__(
                model_fn=_model_fn_esmm, model_dir=model_dir, config=config, params=params,
                warm_start_from=warm_start_from)
