if [ $1 = "tfr_mt" ]; then
    python gen_tfrecored_multi_process_with_gen_mt.py --ds=20250120
elif [ $1 = "mt_ana" ]; then
    python check_recall_distribute.py --range='' --ds=20250126
elif [ $1 = "mt_ana_range" ]; then
    python check_recall_distribute.py --range='20250101,20250102,20250103,20250104,20250105,20250106,20250107,20250108,20250109,20250110,20250111,20250112,20250113,20250114,20250115,20250116,20250117,20250118,20250119'
fi
