if [ $1 = "train" ]; then
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v4 --instance_count=3 --warm_start_from=NEW --train_ds=20250120-20250128 --eval_ds=20250120eval --sample=cn_rec_detail_sample_v30_savana_in_tfr_row_n300
#    sleep 600
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v4 --instance_count=3 --warm_start_from=NEWEST --train_ds=20250129-20250204 --pre_ds=20250120-20250128 --eval_ds=20250120eval --sample=cn_rec_detail_sample_v30_savana_in_tfr_row_n300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v4 --instance_count=3 --warm_start_from=NEWEST --train_ds=20250205-20250217 --pre_ds=20250129-20250204 --eval_ds=20250120eval --sample=cn_rec_detail_sample_v30_savana_in_tfr_row_n300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v4 --instance_count=3 --warm_start_from=NEWEST --train_ds=20250218-20250223 --pre_ds=20250205-20250217 --eval_ds=20250120eval --sample=cn_rec_detail_sample_v30_savana_in_tfr_row_n300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v4 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250225 --pre_ds=20250224 --eval_ds=20250120eval --sample=cn_rec_detail_sample_v30_savana_in_tfr_row_n300
#    sleep 600
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v4 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250226 --pre_ds=20250225 --eval_ds=20250120eval --sample=cn_rec_detail_sample_v30_savana_in_tfr_row_n300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v4 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250227 --pre_ds=20250226 --eval_ds=20250120eval --sample=cn_rec_detail_sample_v30_savana_in_tfr_row_n300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v4 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250228 --pre_ds=20250227 --eval_ds=20250120eval --sample=cn_rec_detail_sample_v30_savana_in_tfr_row_n300
#    sleep 600
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v4 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250301 --pre_ds=20250228 --eval_ds=20250120eval --sample=cn_rec_detail_sample_v30_savana_in_tfr_row_n300
#    sleep 600
    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v4 --instance_count=4 --warm_start_from=NEWEST --train_ds=20250302 --pre_ds=20250227 --eval_ds=20250120eval --sample=cn_rec_detail_sample_v30_savana_in_tfr_row_n300

elif [ $1 = "train_d" ]; then
    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v4 --instance_count=1 --warm_start_from=NEW --train_ds=20250120eval --eval_ds=20250120eval --sample=cn_rec_detail_sample_v30_savana_in_tfr_row_n300

elif [ $1 = "pred_d" ]; then
#    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v4 --ds=20250101-20250119 --proc=1 --sample_num=1
    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v4 --ds=20250120-20250128 --pred_ds=20250129  --proc=1 --sample_num=1
elif [ $1 = "pred" ]; then
    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v4 --ds=20250205-20250217  --pred_ds=20250217
fi
