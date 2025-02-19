if [ $1 = "tfr_mt" ]; then
    python gen_tfrecored_multi_process_with_gen_mt.py --ds=20250218
elif [ $1 = "tfr_mt_range" ]; then
#  20250120,20250121,20250122,20250130,20250131
    python gen_tfrecored_multi_process_with_gen_mt.py --range=20250206,20250207,20250208,20250209,20250210
elif [ $1 = "tfr_mt_range2" ]; then
    python gen_tfrecored_multi_process_with_gen_mt.py --range=20250211,20250212,20250213,20250214
elif [ $1 = "tfr_mt_range3" ]; then
    python gen_tfrecored_multi_process_with_gen_mt.py --range=20250216,20250217,20250218,20250215
elif [ $1 = "tfr_mt_d" ]; then
    python gen_tfrecored_multi_process_with_gen_mt.py --ds=20250120 --sample_num=1 --thread=1
elif [ $1 = "mt_ana" ]; then
    python check_recall_distribute.py --range='' --ds=20250120
elif [ $1 = "mt_ana_pos_idx" ]; then
    python check_recall_distribute_pos_idx.py --range='' --ds=20250120
elif [ $1 = "mt_ana_range_pos_idx" ]; then
    python check_recall_distribute_pos_idx.py
fi
