if [ $1 = "train" ]; then
#    python run_rec_model_sg_di.py  --instance_count=4 --warm_start_from=NEW --train_ds=20241218-20241231
#    sleep 600
    python run_rec_model_sg_di.py  --instance_count=3 --warm_start_from=NEWEST --pre_ds=20250101-20250119 --train_ds=20250120-20250126

elif [ $1 = "train_d" ]; then
    python run_rec_model_sg_di.py  --instance_count=1 --warm_start_from=NEW --train_ds=20250119eval

elif [ $1 = "pred_d" ]; then
#    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v2 --ds=20250101-20250119 --proc=1 --sample_num=1
    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v2 --ds=20241218-20241231 --proc=1 --sample_num=1
elif [ $1 = "pred" ]; then
    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v2 --ds=2050120-20250126
fi
