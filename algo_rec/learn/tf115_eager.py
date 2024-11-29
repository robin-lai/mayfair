import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# dataset = tf.data.TFRecordDataset('./tfrecord/part-00000-827236cb-422b-4758-9f33-265565f1aad3-c000')
# dataset2 =tf.data.TFRecordDataset('./tfrecord/part-00199-827236cb-422b-4758-9f33-265565f1aad3-c000')
filenames = [
    "/home/sagemaker-user/mayfair/algo_rec/data/cn_rec_detail_sample_v1_cvr/ds=20241102/part-00000-84854856-90d1-4bf8-9e52-5032827fdee4-c000",
    "/home/sagemaker-user/mayfair/algo_rec/data/cn_rec_detail_sample_v1_ctr/ds=20241102/part-00000-84854856-90d1-4bf8-9e52-5032827fdee4-c000",
    ]
for file in filenames:
    ds = tf.data.TFRecordDataset(filenames)
    ds.map()
    n2 = 0
    for e in ds:
        n2+=1
print('n2', n2)
