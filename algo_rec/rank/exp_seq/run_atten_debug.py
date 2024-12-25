import argparse
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


def attention_layer(seq_ids, tid_ids,id_type, shape, att_type,seq_len=None, max_len=6):
    with tf.variable_scope("attention_" + id_type):
        print('raw seq_ipt tensor shape:', seq_ids.get_shape())
        print('raw tid_ipt tensor shape:', tid_ids.get_shape())
        seq_ids_hash = tf.string_to_hash_bucket_fast(seq_ids, shape[0])
        tid_ids_hash = tf.string_to_hash_bucket_fast(tid_ids, shape[0])
        with tf.variable_scope("att_" + id_type, reuse=tf.AUTO_REUSE) as name:
            embeddings = tf.get_variable(name="emb_att_" + id_type , dtype=tf.float32,
                                         shape=shape, trainable=True, initializer=tf.random_normal_initializer(seed=10))
        seq_emb = tf.nn.embedding_lookup(embeddings, seq_ids_hash )
        print('seq_emb_shape',seq_emb.get_shape())
        print('seq_emb', seq_emb.numpy().tolist())
        seq_len = seq_emb.get_shape()[1]
        tid_emb = tf.nn.embedding_lookup(embeddings, tid_ids_hash)
        print('tid_emb_shape', tid_emb.get_shape())
        print('tid_emb', tid_emb.numpy().tolist())
        tid_emb_tile = tf.tile(tid_emb, [1, seq_len, 1])
        if att_type == 'din':
            net = tf.concat([seq_emb, tid_emb_tile, seq_emb - tid_emb_tile, seq_emb * tid_emb_tile], axis=-1)
            for layer_id, units in enumerate([4 * shape[1], 2 * shape[1], 8, 1]):
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            score = tf.reshape(net, [-1, 1,  seq_len])
        elif att_type == 'dot':
            score = seq_emb * tid_emb_tile
            score = tf.reduce_mean(score, axis=2)
        print('score_shape', score.get_shape())
        print('score:', score.numpy().tolist())

        if seq_len is not None:
            mask = tf.sequence_mask(seq_len, max_len)
            paddings = tf.zeros_like(score)
            score = tf.where(mask, score, paddings)
        print('score_pad_shape', score.get_shape())
        print('score_pad:', score.numpy().tolist())
        score_softmax = tf.nn.softmax(score)
        print('score_softmax:', score_softmax.numpy().tolist())
        output = tf.matmul(score_softmax, seq_emb)
        print('output:', output.numpy().tolist())
        output_2d = tf.reduce_mean(output, axis=1)
        print('output_2d:', output_2d.numpy().tolist())
        return output_2d

def main(args):
    features = {}
    features['seq_goods_id'] = tf.constant([["1", "1", "3", "", "", ""], ["1", "1", "3", "4", "5", ""]])
    features['goods_id'] = tf.constant([["1"], ["2"]])
    features['seq_len'] = tf.constant([3, 5])


    if args.mask:
        seq_goodsid_input = attention_layer(seq_ids=features['seq_goods_id'], tid_ids=features['goods_id'],
                                        id_type='seq_off_goods_id', shape=[40000, 8], att_type=args.att_type
                                        ,seq_len=features['seq_len'],max_len=args.max_len)
    else:
        seq_goodsid_input = attention_layer(seq_ids=features['seq_goods_id'], tid_ids=features['goods_id'],
                                        id_type='seq_off_goods_id', shape=[40000, 8], att_type=args.att_type)
    print('seq_goodsid_input', seq_goodsid_input.numpy().tolist())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='swing',
        description='swing-args',
        epilog='swing-help')
    parser.add_argument('--att_type',type=str, default='din')
    parser.add_argument('--mask',type=bool, default=False)
    parser.add_argument('--max_len',type=int, default=6)
    args = parser.parse_args()
    main(args)


