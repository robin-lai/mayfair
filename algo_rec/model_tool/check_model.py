import os

import tensorflow as tf

s3_model = 's3://warehouse-algo/rec/prod_model/prod_mtl_seq_on_esmm_v0/ds=20241206/model/'
local_model = './prod_mtl_seq_on_esmm_v0_20241206_model/'
# os.system('aws s3 cp --recursive %s %s'%(s3_model, local_model))
print(tf.train.list_variables(local_model))
print(tf.train.load_variable(local_model, 'attention_seq_on_high_cate_id/att_seq_on_high_cate_id/emb_att_seq_on_high_cate_id'))
