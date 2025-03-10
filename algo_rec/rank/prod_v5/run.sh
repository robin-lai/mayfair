if [ $1 = "train" ]; then
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEW --train_ds=20250215-20250224 --eval_ds=20250304eval
#    sleep 600
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250305 --pre_ds=20250225-20250304  --eval_ds=20250304eval
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250306 --pre_ds=20250305  --eval_ds=20250304eval
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250308 --pre_ds=20250307  --eval_ds=20250304eval
    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250309 --pre_ds=20250308  --eval_ds=20250304eval

elif [ $1 = "train_d" ]; then
    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=1 --warm_start_from=NEW --train_ds=20250304eval --eval_ds=20250304eval 

elif [ $1 = "pred_d" ]; then
    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v5 --ds=20250304eval --pred_ds=20250304eval  --proc=1 --sample_num=1
elif [ $1 = "pred1" ]; then
    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v5 --ds=20250215-20250224  --pred_ds=20250225
elif [ $1 = "pred2" ]; then
    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v5 --ds=20250225-20250304  --pred_ds=20250304
fi
