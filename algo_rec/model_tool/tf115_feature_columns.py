import tensorflow as tf
tf.compat.v1.disable_eager_execution()
goods_id = tf.feature_column.categorical_column_with_identity(
    key='goods_id', num_buckets=100000, default_value=0)
goods_id_seq = tf.feature_column.categorical_column_with_identity(
    key='goods_id_seq', num_buckets=100000, default_value=0)
emb_goods_id = tf.feature_column.shared_embedding_columns([goods_id_seq,goods_id], dimension=4)

goods_id = {"goods_id": tf.constant([[0],[1],[2]]),"goods_id_seq": tf.constant([[0,1,2], [4,5,6],[0,4,5]])}


goods_emb =  tf.feature_column.input_layer(goods_id, emb_goods_id)
# goods_seq_emb = tf.feature_column.input_layer(goods_id_seq, emb_goods_id)
# goods_emb = tf.keras.layers.DenseFeatures(goods_id, emb_goods_id)
# goods_seq_emb = tf.keras.layers.DenseFeatures(goods_id_seq, emb_goods_id)

# goods_emb = tf.compat.v2.keras.layers.DenseFeatures(goods_id, emb_goods_id)
# goods_seq_emb = tf.compat.v2.keras.layers.DenseFeatures(goods_id_seq, emb_goods_id)



print(goods_emb.numpy())
