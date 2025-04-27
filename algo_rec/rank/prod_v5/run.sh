if [ $1 = "train" ]; then
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEW --train_ds=20250215-20250224 --eval_ds=20250304eval
#    sleep 600
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250305 --pre_ds=20250225-20250304  --eval_ds=20250304eval
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250306 --pre_ds=20250305  --eval_ds=20250304eval
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250308 --pre_ds=20250307  --eval_ds=20250304eval
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250309 --pre_ds=20250308  --eval_ds=20250304eval
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=3 --warm_start_from=NEWEST --train_ds=20250310-20250323 --pre_ds=20250309  --eval_ds=20250304eval
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250324 --pre_ds=20250310-20250323  --eval_ds=20250304eval
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250325 --pre_ds=20250324  --eval_ds=20250304eval
#    sleep 600
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250326 --pre_ds=20250325  --eval_ds=20250304eval
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250327 --pre_ds=20250326  --eval_ds=20250304eval
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250328 --pre_ds=20250327  --eval_ds=20250304eval
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250329 --pre_ds=20250328  --eval_ds=20250330
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250330 --pre_ds=20250329  --eval_ds=20250331
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250331 --pre_ds=20250330  --eval_ds=20250401
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250401 --pre_ds=20250331  --eval_ds=20250402
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250402 --pre_ds=20250401  --eval_ds=20250403
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250403 --pre_ds=20250402  --eval_ds=20250404
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250404 --pre_ds=20250403  --eval_ds=20250405
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250405 --pre_ds=20250404  --eval_ds=20250406
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250406 --pre_ds=20250405  --eval_ds=20250407
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250407 --pre_ds=20250406  --eval_ds=20250408
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250408 --pre_ds=20250407  --eval_ds=20250409
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250409 --pre_ds=20250408  --eval_ds=20250410
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250410 --pre_ds=20250409  --eval_ds=20250411
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250411 --pre_ds=20250410  --eval_ds=20250412
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250412 --pre_ds=20250411  --eval_ds=20250413
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250413 --pre_ds=20250412  --eval_ds=20250414
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250414 --pre_ds=20250413  --eval_ds=20250415
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250415 --pre_ds=20250414  --eval_ds=20250416
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250416 --pre_ds=20250415  --eval_ds=20250417
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250417 --pre_ds=20250416  --eval_ds=20250418
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250418 --pre_ds=20250417  --eval_ds=20250419
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250419 --pre_ds=20250418  --eval_ds=20250420
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250420 --pre_ds=20250419  --eval_ds=20250421
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250421 --pre_ds=20250420  --eval_ds=20250421
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250422 --pre_ds=20250421  --eval_ds=20250423
#    sleep 300
#    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250423 --pre_ds=20250422  --eval_ds=20250423
#    sleep 300
    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250424 --pre_ds=20250423  --eval_ds=20250424
    sleep 300
    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250425 --pre_ds=20250424  --eval_ds=20250425
    sleep 300
    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=2 --warm_start_from=NEWEST --train_ds=20250426 --pre_ds=20250425  --eval_ds=20250426
    sleep 300

elif [ $1 = "train_d" ]; then
    python run_rec_model_sg_di.py  --model_name=mtl_seq_esmm_v5 --instance_count=1 --warm_start_from=NEW --train_ds=20250304eval --eval_ds=20250304eval 

elif [ $1 = "pred_d" ]; then
    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v5 --ds=20250304eval --pred_ds=20250304eval  --proc=1 --sample_num=1
elif [ $1 = "pred1" ]; then
    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v5 --ds=20250215-20250224  --pred_ds=20250225
elif [ $1 = "pred2" ]; then
    python predict_tfr_mtl_v2.py --model_name=mtl_seq_esmm_v5 --ds=20250225-20250304  --pred_ds=20250304
fi
