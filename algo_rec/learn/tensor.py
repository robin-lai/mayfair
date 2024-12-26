import tensorflow as tf
tf.compat.v1.enable_eager_execution()

t = tf.constant([1,2,3,4,5])
print(t.numpy().tolist())
# print(t.tolist()) # error

t = tf.constant([[1], [2],[3]])
print(t.get_shape())
tt = tf.reshape(t, shape=[-1, 1])
print(tt.get_shape())
print(tt)
