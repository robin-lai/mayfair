python run_rec_model_sg_di.py  --instance_count=3 --warm_start_from=NEW --train_ds=20241218-20241231
sleep 600
python run_rec_model_sg_di.py  --instance_count=3 --warm_start_from=NEWEST --pre_ds=20241218-20241231 --train_ds=20250101-20250119
