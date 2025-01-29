if [ $1 = "train" ]; then
#    python ./rank/prod_v2/run_rec_model_sg_di.py  --instance_count=4 --warm_start_from=NEW --train_ds=20241218-20241231
#    sleep 600
    python ./rank/prod_v2/run_rec_model_sg_di.py  --instance_count=3 --warm_start_from=NEWEST --pre_ds=20250101-20250119 --train_ds=20250120-20250126

elif [ $1 = "train_d" ]; then
    python ./rank/prod_v2/run_rec_model_sg_di.py  --instance_count=1 --warm_start_from=NEW --train_ds=20250119eval

elif [ $1 = "pred_d" ]; then
#    python ./rank/prod_v2/predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v2 --ds=20250101-20250119 --proc=1 --sample_num=1
    python ./rank/prod_v2/predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v2 --ds=20241218-20241231 --proc=1 --sample_num=1
elif [ $1 = "pred" ]; then
    python ./rank/prod_v2/predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v2 --ds=20250120-20250126 --pred_ds=20250127

elif [ $1 = 'tfr_mt_d' ]; then
    python -m pdb ./data_process/gen_tfrecored_multi_process_with_gen_mt.py --thread=1 --sample_num=1

elif [ $1 = 'tfr_mt' ]; then
    python ./data_process/gen_tfrecored_multi_process_with_gen_mt.py

elif [ $1 = 'deploy' ]; then
    python deploy.py --pipeline=$2  --model_name=mtl_seq_esmm_v2  --edp_version=20250120-20250126
fi
