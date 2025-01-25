if [ $1 = "tfr" ]; then
    python gen_tfrecored_multi_process.py --range=202501

elif [ $1 = "tfr_mt" ]; then
    python run_rec_model_sg_di.py  --instance_count=1 --warm_start_from=NEW --train_ds=20250119eval

fi
