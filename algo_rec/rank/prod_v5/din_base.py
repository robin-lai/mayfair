# encoding:utf-8
import tensorflow as tf
from attention import attention_layer
class DIN_BASE(tf.estimator.Estimator):
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
            cate_cols_emb = params["feature_columns"]["cate_cols_emb"]
            cate_cols_emb_input = tf.feature_column.input_layer(features, cate_cols_emb)
            cate_cols_shared_emb = params["feature_columns"]["cate_cols_share_emb"]
            cate_cols_shared_input = tf.feature_column.input_layer(features, cate_cols_shared_emb)
            numric_cols_emb = params["feature_columns"]["numric_cols_emb"]
            numric_cols_emb_input = tf.feature_column.input_layer(features, numric_cols_emb)

            seq_goodsid_input = attention_layer(seq_ids=features['seq_goods_id'], tid_ids=features['goods_id'],
                                                id_type='seq_off_goods_id', shape=[200000, 32])
            seq_cateid_input = attention_layer(seq_ids=features['seq_cate_id'], tid_ids=features['cate_id'],
                                               id_type='seq_off_cate_id', shape=[2000, 16])

            input_layer = [numric_cols_emb_input, cate_cols_shared_input, cate_cols_emb_input, seq_goodsid_input,
                           seq_cateid_input]
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
                print('modekeys predict:', mode)
                predictions = {
                    'probabilities': prop,
                }
                export_outputs = {
                    'prediction': tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits),
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
            super(DIN_BASE, self).__init__(
                model_fn=_model_fn, model_dir=model_dir, config=config, params=params)
        else:
            super(DIN_BASE, self).__init__(
                model_fn=_model_fn, model_dir=model_dir, config=config, params=params,
                warm_start_from=warm_start_from)