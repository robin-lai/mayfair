import tensorflow as tf
tf.compat.v1.enable_eager_execution()
dataset = tf.data.TFRecordDataset('./tfrecord/part-00000-827236cb-422b-4758-9f33-265565f1aad3-c000')
dataset2 =tf.data.TFRecordDataset('./tfrecord/part-00199-827236cb-422b-4758-9f33-265565f1aad3-c000')
n1 = 0
n2 = 0
for e in dataset:
  n1 += 1
print('n1', n1)
for e in dataset2:
  n2+=1
print('n2', n2)