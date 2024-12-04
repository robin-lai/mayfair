# coding: utf-8

todell_dir = '/home/sagemaker-user/todell/tmp'
deploy_dir = '/home/sagemaker-user/mayfair/algo_rec/deploy/'
deploy_pkg_dir = deploy_dir + 'pkg/'
deploy_tmp_dir = deploy_dir + 'tmp/'
deploy_code_dir = deploy_pkg_dir + 'code/'
deploy_data_dir = deploy_tmp_dir + 'data/'

# item features
fts_item_s3_text_dir = 's3://algo-sg/rec/cn_rec_detail_feature_item_base_for_redis/'
fts_item_local_text_dir = deploy_data_dir + 'cn_rec_detail_feature_item_base_for_redis/'
fts_item_pickle = deploy_pkg_dir + 'item_features.pkl'

fts_user_seq_off_s3 = 's3://algo-sg/rec/cn_rec_detail_feature_user_seq_v2_for_redis/ds=20241201/'
fts_user_seq_off_pickle = deploy_pkg_dir + 'user_seq_off_features.pkl'


# config
# s3_model = 's3://warehouse-algo/rec/prod_model/prod_ctr_seq_off_din_v0/ds=20241107/model/'
s3_model = 's3://warehouse-algo/rec/test_model/predict_test3/ds=20241112/model'
model_local = '~/mayfair/algo_rec/rank/exp/model_seq_nohead_1day_1128/1732783231/'
# s3_model_online = 's3://algo-sg/rec/model_online/'
s3_model_online = 's3://algo-rec/rec/model_online/'
tar_name = 'model_seq_off_1204_test_5.tar.gz'
s3_model_online_tar_file = s3_model_online + tar_name
endpoint = 'ctr-model-seq-off-1204-5'

s3_sp_pt_dir = "s3://warehouse-algo/rec/cn_rec_detail_sample_v1/"
s3_sp_pt_dir_key = "rec/cn_rec_detail_sample_v1/"
s3_sp_tfr_dir = "s3://warehouse-algo/rec/cn_rec_detail_sample_v1_tfr-all/"
