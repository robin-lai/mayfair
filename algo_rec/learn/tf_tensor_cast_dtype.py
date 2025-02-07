import tensorflow as tf

# 创建一个 int64 类型的张量
tensor_int64 = tf.constant([1, 2, 3], dtype=tf.int64)

# 将张量类型转换为 float32
tensor_float32 = tf.cast(tensor_int64, dtype=tf.float32)

# 启动一个会话来运行计算图并获取结果
with tf.Session() as sess:
    result = sess.run(tensor_float32)
    print(result)