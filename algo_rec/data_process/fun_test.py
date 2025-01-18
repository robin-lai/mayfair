from gen_tfrecored_multi_process_with_gen_mt import get_hot_i2leaf,get_site_hot

hot_i2leaf = get_hot_i2leaf('s3://warehouse-algo/rec/cn_rec_detail_recall_main_leaf2i_ds/ds=20240113/')
site_hot = get_site_hot('s3://warehouse-algo/rec/cn_rec_detail_recall_site_hot/ds=20240113/')