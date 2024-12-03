nohup python run_rec_model_sg_di.py --mode=infer --eval_ds=20241111 --pre_ds=20241110 --ds=20241111 --train_ds=20241110 --instance_count=1 --task=cvr --model_name=prod_cvr_seq_off_din_v0 > cvr_predict.log 2>&1 &
# for test
nohup python run_rec_model_sg_di.py  --task=cvr --model_dir=test_model --model_name=predict_test --mode=train --warm_start_from=NEW --train_ds=20241112 --eval_ds=20241112 --pre_ds=20241112 --ds=20241112  --instance_count=1  > cvr_predict.log 2>&1 &
