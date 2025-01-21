# encoding:utf-8
import tensorflow as tf

def attention_layer(seq_ids, tid_ids, id_type, shape):
    with tf.variable_scope("attention_" + id_type):
        print('raw seq_ipt tensor shape:', seq_ids.get_shape())
        print('raw tid_ipt tensor shape:', tid_ids.get_shape())
        seq_ids_hash = tf.string_to_hash_bucket_fast(seq_ids, shape[0])
        tid_ids_hash = tf.string_to_hash_bucket_fast(tid_ids, shape[0])
        with tf.variable_scope("att_" + id_type, reuse=tf.AUTO_REUSE) as name:
            embeddings = tf.get_variable(name="emb_att_" + id_type, dtype=tf.float32,
                                         shape=shape, trainable=True, initializer=tf.glorot_uniform_initializer())
        seq_emb = tf.nn.embedding_lookup(embeddings, seq_ids_hash)
        seq_len = seq_emb.get_shape()[1]
        print('seq_emb', seq_emb.get_shape())
        tid_emb = tf.nn.embedding_lookup(embeddings, tid_ids_hash)
        print('tid_emb', tid_emb.get_shape())
        tid_emb_tile = tf.tile(tid_emb, [1, seq_len, 1])
        net = tf.concat([seq_emb, tid_emb_tile, seq_emb - tid_emb_tile, seq_emb * tid_emb_tile], axis=-1)
        for layer_id, units in enumerate([4 * shape[1], 2 * shape[1], 8, 1]):
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        score = tf.reshape(net, [-1, 1, seq_len])
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


def attention_layer_mask_v2(seq_ids_high, seq_ids_low, tid_ids, id_type, shape, att_type, seq_len_actual_high=None,
                            seq_len_actual_low=None, max_len=20,
                            initialize='normal'):
    def din_fun(seq_emb, tid_emb, seq_len_actual):
        seq_len = seq_emb.get_shape()[1]
        tid_emb_tile = tf.tile(tid_emb, [1, seq_len, 1])
        if att_type == 'net':
            net = tf.concat([seq_emb, tid_emb_tile, seq_emb - tid_emb_tile, seq_emb * tid_emb_tile], axis=-1)
            for layer_id, units in enumerate([4 * shape[1], 2 * shape[1], 8, 1]):
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            score = tf.reshape(net, [-1, 1, seq_len])
        elif att_type == 'dot':
            score = seq_emb * tid_emb_tile
            score = tf.reduce_mean(score, axis=2)
        print('score_shape', score.get_shape())
        # print('score:', score.numpy().tolist())

        paddings = tf.zeros_like(score)
        if seq_len_actual is not None:
            mask = tf.sequence_mask(seq_len_actual, max_len)
            mask = tf.reshape(mask, [-1, max_len])
            score = tf.where(mask, score, paddings)
        print('score_pad_shape', score.get_shape())
        # print('score_pad:', score.numpy().tolist())
        score_softmax = tf.nn.softmax(score)
        # print('score_softmax:', score_softmax.numpy().tolist())
        if seq_len_actual is not None:
            score_softmax = tf.where(mask, score_softmax, paddings)
        # print('score_softmax_pad:', score_softmax.numpy().tolist())
        # output = tf.matmul(score_softmax, seq_emb) # 3,6 matmul 3,6,8 = 3,3,8
        score_softmax = tf.expand_dims(score_softmax, axis=-1)
        print('score_softmax_pad_expand_shape:', score_softmax.get_shape())
        # print('score_softmax_pad_expand:', score_softmax.numpy().tolist())
        output = score_softmax * seq_emb  # 3,6 matmul 3,6,8 = 3,3,8
        # print('output:', output.numpy().tolist())
        output_2d = tf.reduce_mean(output, axis=1)
        # print('output_2d:', output_2d.numpy().tolist())
        return output_2d

    with tf.variable_scope("attention_" + id_type):
        print('raw seq_high_ipt tensor shape:', seq_ids_high.get_shape())
        print('raw seq_low_ipt tensor shape:', seq_ids_low.get_shape())
        print('raw tid_ipt tensor shape:', tid_ids.get_shape())
        seq_ids_high_hash = tf.string_to_hash_bucket_fast(seq_ids_high, shape[0])
        seq_ids_low_hash = tf.string_to_hash_bucket_fast(seq_ids_low, shape[0])
        tid_ids_hash = tf.string_to_hash_bucket_fast(tid_ids, shape[0])
        with tf.variable_scope("att_" + id_type, reuse=tf.AUTO_REUSE) as name:
            if initialize == 'zero':
                embeddings = tf.get_variable(name="emb_att_" + id_type, dtype=tf.float32,
                                             shape=shape, trainable=True, initializer=tf.zeros_initializer())
            else:
                embeddings = tf.get_variable(name="emb_att_" + id_type, dtype=tf.float32,
                                             shape=shape, trainable=True,
                                             initializer=tf.random_normal_initializer(seed=10))
        tid_emb = tf.nn.embedding_lookup(embeddings, tid_ids_hash)
        print('tid_emb_shape', tid_emb.get_shape())

        seq_high_emb = tf.nn.embedding_lookup(embeddings, seq_ids_high_hash)
        print('seq_high_emb_shape', seq_high_emb.get_shape())
        # print('seq_emb', seq_emb.numpy().tolist())
        seq_high_output = din_fun(seq_high_emb, tid_emb, seq_len_actual_high)

        seq_low_emb = tf.nn.embedding_lookup(embeddings, seq_ids_low_hash)
        print('seq_low_emb_shape', seq_low_emb.get_shape())
        seq_low_output = din_fun(seq_low_emb, tid_emb, seq_len_actual_low)
        return [seq_high_output, seq_low_output]


def attention_layer_mask(seq_ids, tid_ids, id_type, shape, att_type, seq_len_actual=None, max_len=20,
                         initialize='normal'):
    with tf.variable_scope("attention_" + id_type):
        print('raw seq_ipt tensor shape:', seq_ids.get_shape())
        print('raw tid_ipt tensor shape:', tid_ids.get_shape())
        seq_ids_hash = tf.string_to_hash_bucket_fast(seq_ids, shape[0])
        tid_ids_hash = tf.string_to_hash_bucket_fast(tid_ids, shape[0])
        with tf.variable_scope("att_" + id_type, reuse=tf.AUTO_REUSE) as name:
            if initialize == 'zero':
                embeddings = tf.get_variable(name="emb_att_" + id_type, dtype=tf.float32,
                                             shape=shape, trainable=True, initializer=tf.zeros_initializer())
            else:
                embeddings = tf.get_variable(name="emb_att_" + id_type, dtype=tf.float32,
                                             shape=shape, trainable=True,
                                             initializer=tf.random_normal_initializer(seed=10))
        seq_emb = tf.nn.embedding_lookup(embeddings, seq_ids_hash)
        print('seq_emb_shape', seq_emb.get_shape())
        # print('seq_emb', seq_emb.numpy().tolist())
        seq_len = seq_emb.get_shape()[1]
        tid_emb = tf.nn.embedding_lookup(embeddings, tid_ids_hash)
        print('tid_emb_shape', tid_emb.get_shape())
        # print('tid_emb', tid_emb.numpy().tolist())
        tid_emb_tile = tf.tile(tid_emb, [1, seq_len, 1])
        if att_type == 'net':
            net = tf.concat([seq_emb, tid_emb_tile, seq_emb - tid_emb_tile, seq_emb * tid_emb_tile], axis=-1)
            for layer_id, units in enumerate([4 * shape[1], 2 * shape[1], 8, 1]):
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            score = tf.reshape(net, [-1, 1, seq_len])
        elif att_type == 'dot':
            score = seq_emb * tid_emb_tile
            score = tf.reduce_mean(score, axis=2)
        print('score_shape', score.get_shape())
        # print('score:', score.numpy().tolist())

        paddings = tf.zeros_like(score)
        if seq_len_actual is not None:
            mask = tf.sequence_mask(seq_len_actual, max_len)
            mask = tf.reshape(mask, [-1, max_len])
            score = tf.where(mask, score, paddings)
        print('score_pad_shape', score.get_shape())
        # print('score_pad:', score.numpy().tolist())
        score_softmax = tf.nn.softmax(score)
        # print('score_softmax:', score_softmax.numpy().tolist())
        if seq_len_actual is not None:
            score_softmax = tf.where(mask, score_softmax, paddings)
        # print('score_softmax_pad:', score_softmax.numpy().tolist())
        # output = tf.matmul(score_softmax, seq_emb) # 3,6 matmul 3,6,8 = 3,3,8
        score_softmax = tf.expand_dims(score_softmax, axis=-1)
        print('score_softmax_pad_expand_shape:', score_softmax.get_shape())
        # print('score_softmax_pad_expand:', score_softmax.numpy().tolist())
        output = score_softmax * seq_emb  # 3,6 matmul 3,6,8 = 3,3,8
        # print('output:', output.numpy().tolist())
        output_2d = tf.reduce_mean(output, axis=1)
        # print('output_2d:', output_2d.numpy().tolist())
        return output_2d