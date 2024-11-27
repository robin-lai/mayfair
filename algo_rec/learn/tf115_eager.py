import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# dataset = tf.data.TFRecordDataset('./tfrecord/part-00000-827236cb-422b-4758-9f33-265565f1aad3-c000')
# dataset2 =tf.data.TFRecordDataset('./tfrecord/part-00199-827236cb-422b-4758-9f33-265565f1aad3-c000')
filenames = ["/home/sagemaker-user/mayfair/algo_rec/rank/exp/cn_rec_detail_sample_v1_tfr-all/ds=20241112/part-00003-827236cb-422b-4758-9f33-265565f1aad3-c000",
             "/home/sagemaker-user/mayfair/algo_rec/rank/exp/cn_rec_detail_sample_v1_tfr-all/ds=20241112/part-00002-827236cb-422b-4758-9f33-265565f1aad3-c000"]
dataset3 = tf.data.TFRecordDataset(filenames)
n1 = 0
n2 = 0
for e in dataset3:
  n1 += 1
print('n1', n1)
# for e in dataset2:
#   n2+=1
# print('n2', n2)