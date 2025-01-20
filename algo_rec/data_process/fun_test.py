from gen_tfrecored_multi_process_with_gen_mt import get_hot_i2leaf,get_site_hot, get_u2cart_wish,get_i2i

# hot_i2leaf = get_hot_i2leaf('s3://warehouse-algo/rec/recall/cn_rec_detail_recall_main_leaf2i_ds/ds=20250113/')
i2i = get_i2i(10, 'rec/recall/cn_rec_detail_recall_i2i_for_redis/item_user_debias_20250117_1.0_0.7_0.5/', 'swing_rec_Savana_IN_part_%s')
# site_hot = get_site_hot('s3://warehouse-algo/rec/recall/cn_rec_detail_recall_site_hot/ds=20250113/')
# u2cart_wish = get_u2cart_wish('s3://warehouse-algo/rec/recall/cn_rec_detail_recall_wish_cart2i/ds=20250113/')
print('test end')