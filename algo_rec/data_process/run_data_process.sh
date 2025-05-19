if [ $1 = "tfr_mt" ]; then
    python gen_tfrecored_multi_process_with_gen_mt.py --ds=
elif [ $1 = "tfr_mt_ss" ]; then
    python gen_tfrecored_multi_process_with_gen_mt_sample_select.py --ds=$2
elif [ $1 = "tfr_mt_range" ]; then
    python gen_tfrecored_multi_process_with_gen_mt.py --range=20250324,20250325,20250326
elif [ $1 = "tfr_mt_range_ss" ]; then
    python gen_tfrecored_multi_process_with_gen_mt_sample_select.py --range=20250424,20250425,20250426
elif [ $1 = "tfr_mt_range2_ss" ]; then
    python gen_tfrecored_multi_process_with_gen_mt_sample_select.py --range=20250411,20250401,20250402,20250403
elif [ $1 = "tfr_mt_range3_ss" ]; then
    python gen_tfrecored_multi_process_with_gen_mt_sample_select.py --range=20250326,20250327,20250328,20250421
elif [ $1 = "tfr_mt_d" ]; then
    python -m pdb gen_tfrecored_multi_process_with_gen_mt_sample_select.py --ds=20250215 --sample_num=1 --thread=1
elif [ $1 = "tfr_mt_d_ss" ]; then
    python gen_tfrecored_multi_process_with_gen_mt_sample_select.py --ds=20250304 --sample_num=1 --thread=1
elif [ $1 = "mt_ana" ]; then
    python check_recall_distribute.py --range='' --ds=20250120
elif [ $1 = "mt_ana_pos_idx" ]; then
    python check_recall_distribute_pos_idx.py --range='' --ds=20250120
elif [ $1 = "mt_ana_range_pos_idx" ]; then
    python check_recall_distribute_pos_idx.py
fi