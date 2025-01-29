if [ $1 = "tfr_mt" ]; then
    python gen_tfrecored_multi_process_with_gen_mt.py
elif [ $1 = "other" ]; then
    python run_rec_model_sg_di.py  --instance_count=1 --warm_start_from=NEW --train_ds=20250119eval

fi
