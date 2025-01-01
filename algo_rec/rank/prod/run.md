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

# v1_mask
nohup python run_rec_model_sg_di_prod_v1_mask.py --model_name=prod_mtl_seq_on_esmm_v1_mask  --warm_start_from=NEW  --train_ds=20241202-20241209 --instance_count=4   > run_seq_on_mask.log 2>&1 &
nd merge score cost: -6.437301635742188e-05
end write2table cost: 22.61893892288208
****************************************
model_name: prod_mtl_seq_on_esmm_v1_mask
model_version: /ds=20241202-20241209/model/1735200130
tfr_s3: rec/cn_rec_detail_sample_v10_tfr/ds=20241210/
N: 2697072 avg_pred_cvr: 0.005452963238471872 avg_label_pay: 0.006023940035712803
cvr-auc: 0.6692640859822552
compute cvr-auc cost: 3.975188970565796
N: 23885532 avg_pred_ctr: 0.11377832367715975 avg_label_clk: 0.11291655551151215
ctr-auc: 0.5839594195351057
compute ctr-auc cost: 28.422733068466187

N: 23885532 label_mean: 0.11291655551151215 pred_mean: 0.11377832367715975 auc-all-ctr: 0.5839594195351053
N: 2697072 label_mean: 0.006023940035712803 pred_mean: 0.005452963238471872 auc-all-ctr: 0.6692640859822552
uuid num: 366036
recid num: 4372847
none_auc num 98459 of all uuid:366036
uuid num:267577 have auc
type:u-gauc 0.49855814157573963
type:u-gauc percentle: [0.27610935 0.375      0.43       0.5        0.52777778 0.56643357
 0.62121212 0.72447758 1.        ]
none_auc num 2509807 of all uuid:4372847
uuid num:1863040 have auc
type:q-gauc 0.48995064279993655
type:q-gauc percentle: [0.         0.16666667 0.3        0.5        0.57142857 0.66666667
 0.8        1.         1.        ]


# v1_mask_zero
nohup python -u run_rec_model_sg_di_prod_v1_mask.py --model_name=prod_mtl_seq_on_esmm_v1_mask_zero --initialize=zero --warm_start_from=NEW  --train_ds=20241202-20241209 --instance_count=4   > run_seq_on_mask_zero.log 2>&1 &

nohup python predict_tfr_mtl_v1_mask_time_opt_with_file.py --model_name=prod_mtl_seq_on_esmm_v1_mask_zero --tfr_s3=rec/cn_rec_detail_sample_v10_tfr/ds=20241210/  --model_version=/ds=20241202-20241209/model/1735233838 > runv2_savana_in.log 2>&1 &
* test:
  python predict_tfr_mtl_v1_mask_time_opt_with_file.py --model_name=prod_mtl_seq_on_esmm_v1_mask_zero  --tfr_s3=rec/cn_rec_detail_sample_v10_tfr/ds=20241210/  --model_version=/ds=20241202-20241209/model/1735233838 --sample_num=1 --proc=1
****************************************
model_name: prod_mtl_seq_on_esmm_v1_mask_zero
model_version: /ds=20241202-20241209/model/1735233838
tfr_s3: rec/cn_rec_detail_sample_v10_tfr/ds=20241210/
N: 2697072 avg_pred_cvr: 0.0058745611610977475 avg_label_pay: 0.006023940035712803
cvr-auc: 0.6653962879454792
compute cvr-auc cost: 3.7314655780792236
N: 23885532 avg_pred_ctr: 0.11495262201891711 avg_label_clk: 0.11291655551151215
ctr-auc: 0.5838542805442902
compute ctr-auc cost: 27.98888325691223

N: 23885532 label_mean: 0.11291655551151215 pred_mean: 0.11495262201891711 auc-all-ctr: 0.5838542805442898
N: 2697072 label_mean: 0.006023940035712803 pred_mean: 0.0058745611610977475 auc-all-ctr: 0.6653962879454793
uuid num: 366036
recid num: 4372847
none_auc num 98459 of all u-gauc :366036
u-gauc num:267577 have auc
type:u-gauc 0.49620909718372347
type:u-gauc percentle: [0.27272727 0.375      0.42857143 0.49707602 0.52513464 0.56333373
 0.61904762 0.72222222 1.        ]
none_auc num 2509807 of all q-gauc :4372847
q-gauc num:1863040 have auc
type:q-gauc 0.4884779313963087
type:q-gauc percentle: [0.         0.16666667 0.3        0.5        0.5625     0.66666667
 0.8        1.         1.        ]




# prod_ctr_seq_on_din_v20_mask_savana_in 
* nohup python -u run_rec_model_sg_di_prod_v1_mask.py --train_ds=20241202-20241209 --warm_start_from=NEW --task=ctr --model_name=prod_ctr_seq_on_din_v20_mask_savana_in --instance_count=4 > v201228.log 2>&1
nohup python -u run_rec_model_sg_di_prod_v1_mask.py --train_ds=20241210-20241216 --warm_start_from=NEWEST --task=ctr --model_name=prod_ctr_seq_on_din_v20_mask_savana_in --instance_count=4 > v201228.log 2>&1
* predict: --
* nohup python predict_tfr_mtl_v1_mask_time_opt_with_file.py --model_name=prod_ctr_seq_on_din_v20_mask_savana_in   --model_version=/ds=20241202-20241209/model/1735387684 > runv2_savana_in.log 2>&1 &
* test: 
python predict_tfr_mtl_v1_mask_time_opt_with_file.py --model_name=prod_ctr_seq_on_din_v20_mask_savana_in   --model_version=/ds=20241202-20241209/model/1735387684 --sample_num=1 --proc=1
model_name: prod_ctr_seq_on_din_v20_mask_savana_in
model_version: /ds=20241202-20241209/model/1735387684
tfr_s3: rec/cn_rec_detail_sample_v20_savana_in_tfr/ds=20241210/
N: 2488789 avg_pred_cvr: 0.12254157575338512 avg_label_pay: 0.006350076282079357
cvr-auc: 0.4709768482468581
compute cvr-auc cost: 3.290285587310791
N: 22172843 avg_pred_ctr: 0.11678284083071333 avg_label_clk: 0.11224492050929148
ctr-auc: 0.5699700505602705
compute ctr-auc cost: 24.973762273788452

N: 22172843 label_mean: 0.11224492050929148 pred_mean: 0.11678284083071333 auc-all-ctr: 0.5699700505602704
N: 2488789 label_mean: 0.006350076282079357 pred_mean: 0.12254157575338512 auc-all-ctr: 0.47097684824685815
uuid num: 338640
recid num: 3991490
none_auc num 91865 of all u-gauc :338640
u-gauc num:246775 have auc
type:u-gauc 0.5598931603191012
type:u-gauc percentle: [0.33333333 0.4375     0.49726447 0.56402439 0.59808779 0.6368797
 0.69163557 0.8        1.        ]
none_auc num 2301627 of all q-gauc :3991490
q-gauc num:1689863 have auc
type:q-gauc 0.5696350461462475
type:q-gauc percentle: [0.         0.25       0.375      0.6        0.6875     0.80952381
 1.         1.         1.        ]

# prod_ctr_seq_on_din_v20_mask_savana_in_fix
* nohup python -u run_rec_model_sg_di_prod_v1_mask.py  --warm_start_from=NEW --task=ctr --model_name=prod_ctr_seq_on_din_v20_mask_savana_in_fix --instance_count=3 > 1230_ctr.log 2>&1 &
nohup python -u run_rec_model_sg_di_prod_v1_mask.py --train_ds=20241210-20241216 --warm_start_from=NEWEST --task=ctr --model_name=prod_ctr_seq_on_din_v20_mask_savana_in --instance_count=4 > v201228.log 2>&1
* predict: --
* nohup python predict_tfr_mtl_v1_mask_time_opt_with_file.py --model_name=prod_ctr_seq_on_din_v20_mask_savana_in   --model_version=/ds=20241202-20241209/model/1735387684 > runv2_savana_in.log 2>&1 &
* test: 

# prod_mtl_seq_on_esmm_v20_mask_savana_in
* nohup python -u run_rec_model_sg_di_prod_v1_mask.py --train_ds=20241202-20241209 --warm_start_from=NEW --task=mtl --model_name=prod_mtl_seq_on_esmm_v20_mask_savana_in --instance_count=3 > v201228.log 2>&1
  nohup python -u run_rec_model_sg_di_prod_v1_mask.py --train_ds=20241210-20241216 --warm_start_from=NEWEST --task=ctr --model_name=prod_ctr_seq_on_din_v20_mask_savana_in --instance_count=4 > v201228.log 2>&1
* predict: --
* nohup python predict_tfr_mtl_v1_mask_time_opt_with_file.py --model_name=prod_ctr_seq_on_din_v20_mask_savana_in   --model_version=/ds=20241202-20241209/model/1735387684 > runv2_savana_in.log 2>&1 &
* test:
  python predict_tfr_mtl_v1_mask_time_opt_with_file.py --model_name=prod_ctr_seq_on_din_v20_mask_savana_in   --model_version=/ds=20241202-20241209/model/1735387684 --sample_num=1 --proc=1

[//]: # (s3://warehouse-algo/rec/prod_model/prod_mtl_seq_on_esmm_v20_mask_savana_in_fix/ds=20241202-20241209/model/1735496971/)
# prod_mtl_seq_on_esmm_v20_mask_savana_in_fix -fix sample
* nohup python -u run_rec_model_sg_di_prod_v1_mask.py  --warm_start_from=NEW --task=mtl --model_name=prod_mtl_seq_on_esmm_v20_mask_savana_in_fix --instance_count=3 > 1230.log 2>&1 &
  nohup python -u run_rec_model_sg_di_prod_v1_mask.py --train_ds=20241210-20241216 --pre_ds=20241202-20241209  --warm_start_from=NEWEST --task=mtl --model_name=prod_mtl_seq_on_esmm_v20_mask_savana_in_fix --instance_count=4 > v0101.log 2>&1 &
* predict: --
* nohup python predict_tfr_mtl_v1_mask_time_opt_with_file.py --model_name=prod_mtl_seq_on_esmm_v20_mask_savana_in_fix   --model_version=/ds=20241202-20241209/model/1735496971 > runv2_savana_in.log 2>&1 &
* test:
  python predict_tfr_mtl_v1_mask_time_opt_with_file.py --model_name=prod_mtl_seq_on_esmm_v20_mask_savana_in_fix   --model_version=/ds=20241202-20241209/model/1735496971 --sample_num=1 --proc=1 
model_name: prod_mtl_seq_on_esmm_v20_mask_savana_in_fix
model_version: /ds=20241202-20241209/model/1735496971
tfr_s3: rec/cn_rec_detail_sample_v20_savana_in_tfr/ds=20241210/
N: 2488789 avg_pred_cvr: 0.0070265482366829194 avg_label_pay: 0.006350076282079357
cvr-auc: 0.5146998798042269
compute cvr-auc cost: 4.0397186279296875
N: 22172843 avg_pred_ctr: 0.11627199131462214 avg_label_clk: 0.11224492050929148
ctr-auc: 0.5782970293469318
compute ctr-auc cost: 26.08153510093689
* 
N: 22172843 label_mean: 0.11224492050929148 pred_mean: 0.11627199131462214 auc-all-ctr: 0.578297029346932
N: 2488789 label_mean: 0.006350076282079357 pred_mean: 0.0070265482366829194 auc-all-ctr: 0.514699879804227
uuid num: 338640
recid num: 3991490
none_auc num 91865 of all u-gauc :338640
u-gauc num:246775 have auc
type:u-gauc 0.5621014388242207
type:u-gauc percentle: [0.33333333 0.4375     0.5        0.56666667 0.6        0.64084507
 0.69545948 0.8        1.        ]
none_auc num 2301627 of all q-gauc :3991490
q-gauc num:1689863 have auc
type:q-gauc 0.566905920633207
type:q-gauc percentle: [0.         0.25       0.375      0.59259259 0.66666667 0.8
 1.         1.         1.        ]


# predict
nohup python predict_tfr_mtl_v1_mask_time_opt_with_file.py --model_name=prod_mtl_seq_on_esmm_v20_mask_savana_in_fix   --model_version=/ds=20241202-20241209/model/1735496971/ --tfr_s3=rec/cn_rec_detail_sample_v20_savana_in_tfr/ds=20241210/ > run.log 2>&1 &
nohup python predict_tfr_mtl_v1_mask_time_opt_with_file.py --model_name=prod_mtl_seq_on_esmm_v20_mask_savana_in_fix   --model_version=/ds=20241210-20241216/model/1735706837/ --tfr_s3=rec/cn_rec_detail_sample_v20_savana_in_tfr/ds=20241217/ > run.log 2>&1 &

nohup python predict_tfr_mtl_v1_mask_time_opt_with_file.py --model_name=prod_mtl_seq_on_esmm_v1_mask_zero             --model_version=/ds=20241202-20241209/model/1735233838/ --tfr_s3=rec/cn_rec_detail_sample_v10_tfr/ds=20241210/   > run.log 2>&1 &


                                            