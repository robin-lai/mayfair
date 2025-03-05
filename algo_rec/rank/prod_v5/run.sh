if [ $1 = "train" ]; then
    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=3 --warm_start_from=NEW --train_ds=20250120-20250128 --eval_ds=20250304eval 

elif [ $1 = "train_d" ]; then
    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=1 --warm_start_from=NEW --train_ds=20250304eval --eval_ds=20250304eval 

elif [ $1 = "pred_d" ]; then
    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v5 --ds=20250304eval --pred_ds=20250304eval  --proc=1 --sample_num=1
elif [ $1 = "pred" ]; then
    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v5 --ds=20250205-20250217  --pred_ds=20250217
fi
