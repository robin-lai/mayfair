import tensorflow as tf

# 创建一个 int64 类型的张量
tensor_int64 = tf.constant([1, 2, 3], dtype=tf.int64)

# 将张量的类型从 int64 转换为 float32
tensor_float32 = tf.cast(tensor_int64, dtype=tf.float32)

# 打印结果
print("Original tensor (int64):", tensor_int64)
print("Converted tensor (float32):", tensor_float32)