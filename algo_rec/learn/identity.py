import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.int32, shape=(None, 3))
y = tf.identity(x)

with tf.Session() as sess:
    feed_dict = dict()
    feed_dict[x] = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    input = sess.run([y], feed_dict=feed_dict)