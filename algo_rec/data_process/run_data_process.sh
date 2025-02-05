if [ $1 = "tfr_mt" ]; then
    python gen_tfrecored_multi_process_with_gen_mt.py
elif [ $1 = "mt_ana" ]; then
    python check_recall_distribute.py --range='' --ds=20250126
fi
