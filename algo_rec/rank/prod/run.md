# infer
nohup python run_rec_model_sg_di.py  --task=cvr --model_dir=test_model --model_name=predict_test3 --mode=infer --warm_start_from=NEWEST --eval_ds=20241112 --pre_ds=20241112 --ds=20241112 --train_ds=20241112 --instance_count=1 > cvr_predict.log 2>&1 &
# for test
nohup python run_rec_model_sg_di.py  --task=cvr --model_dir=test_model --model_name=predict_test --mode=train --warm_start_from=NEW --train_ds=20241112 --eval_ds=20241112 --pre_ds=20241112 --ds=20241112  --instance_count=1  > cvr_predict.log 2>&1 &
# mtl train
nohup python run_rec_model_sg_di.py  --task=mtl --model_dir=test_model --model_name=test_mtl_seq_all_esmm_v0 --mode=train --warm_start_from=NEW --sample=cn_rec_detail_sample_v10_ctr  --train_ds=20241202 --eval_ds=20241202 --pre_ds=20241202 --ds=20241202  --instance_count=1  > esmm_predict.log 2>&1 &

nohup python run_rec_model_sg_di.py  --task=mtl --model_dir=test_model --model_name=test_mtl_seq_all_esmm_1203 --mode=train --warm_start_from=NEW --sample=cn_rec_detail_sample_v10_ctr  --train_ds=20241203 --eval_ds=20241202test --pre_ds=20241203 --ds=20241203  --instance_count=4  > esmm_predict.log 2>&1 &

nohup python run_rec_model_sg_di.py  --task=mtl --model_dir=prod_model --model_name=prod_mtl_seq_all_esmm_v0 --mode=train --warm_start_from=NEWEST --sample=cn_rec_detail_sample_v10_ctr  --range=20241203,20241204,20241205 --eval_ds=20241202test  --instance_count=4  > esmm_predict.log 2>&1 &

nohup python run_rec_model_sg_di.py  --model_dir=prod_model --mode=train --warm_start_from=NEW  --train_ds=20241203  --instance_count=4  > esmm_predict.log 2>&1 &

nohup python run_rec_model_sg_di.py  --model_dir=test_model --mode=train --warm_start_from=NEW  --train_ds=20241202test  --instance_count=1  > esmm_1209.log 2>&1 &
# default-mtl-train-NEWEST, 只换样本日期的训练方式
nohup python run_rec_model_sg_di.py    --range=20241203,20241204,20241205,20241206  --instance_count=4  > esmm_train.log 2>&1 &





# predict_tfr_mtl.py
nohup python predict_tfr_mtl.py --sample_num=10 --model_name=prod_mtl_seq_on_esmm_v0  --model_version=/ds=20241203/model/1733720468/  --tfr_s3=rec/cn_rec_detail_sample_v10_ctr/ds=20241203/  --proc=3 > esmm_predict.log 2>&1 &
nohup python predict_tfr_mtl.py --sample_num=10 --model_name=prod_mtl_seq_on_esmm_v0 --tfr_s3=rec/cn_rec_detail_sample_v10_ctr/ds=20241203/  --proc=3 > esmm_predict.log 2>&1 &
nohup python predict_tfr_mtl.py --sample_num=1 --model_name=prod_mtl_seq_on_esmm_v0 --tfr_s3=rec/cn_rec_detail_sample_v10_ctr/ds=20241203/  --proc=1 > esmm_predict.log 2>&1 &
nohup python predict_tfr_mtl.py --model_name=prod_mtl_seq_on_esmm_v0 --tfr_s3=rec/cn_rec_detail_sample_v10_ctr/ds=20241203/  --proc=15 > esmm_predict.log 2>&1 &

nohup python predict_tfr_mtl.py --model_name=prod_mtl_seq_on_esmm_v1 --model_version=/ds=20241202-20241209/model/1734033700  --tfr_s3=rec/cn_rec_detail_sample_v10_tfr/ds=20241210/  --proc=15 > esmm_predict.log 2>&1 &

# deploy
python deploy.py --pipeline=pkg --edp_version=v7 --tar_name=prod_mtl_seq_on_esmm_v0_v7.tar.gz --region=in
python deploy.py --pipeline=edp --edp_version=v7 --tar_name=prod_mtl_seq_on_esmm_v0_v7.tar.gz --region=in

# v2_savana_in
nohup python run_rec_model_sg_di_prod_v1.py --model_name=prod_mtl_seq_on_esmm_v2_savana_in  --warm_start_from=NEW  --train_ds=20241202-20241209 --instance_count=4 --site_code=Savana_IN  > run_savana_in.log 2>&1 &
nohup python predict_tfr_mtl_v1_time_opt_with_file.py --model_name=prod_mtl_seq_on_esmm_v2_savana_in  --site_code=Savana_IN  --model_version=/ds=20241202-20241209/model/1734534246 > runv2_savana_in.log 2>&1 &
model_name: prod_mtl_seq_on_esmm_v2_savana_in
model_version: /ds=20241202-20241209/model/1734534246
tfr_s3: rec/cn_rec_detail_sample_v10_tfr/ds=20241210/
N: 2544598 avg_pred_cvr: 0.005711605207791523 avg_label_pay: 0.006242243372037548
cvr-auc: 0.6643815645093698
compute cvr-auc cost: 3.601358413696289
N: 22221789 avg_pred_ctr: 0.11485489584755129 avg_label_clk: 0.11450914235573023
ctr-auc: 0.5873199451961778
compute ctr-auc cost: 26.226640462875366
                                            