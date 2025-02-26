if [ $1 = "pkg" ]; then
    python deploy.py --pipeline=pkg  --model_name=mtl_seq_esmm_v4  --edp_version=20250224 --region=in --debug_v='' --ds=20250224
elif [ $1 = "edp" ]; then
    python deploy.py --pipeline=edp  --model_name=mtl_seq_esmm_v4  --edp_version=20250224 --region=in --debug_v='' --ds=20250224
elif [ $1 = "local" ]; then # change base_data_dir on inference.py
    python inference.py
elif [ $1 = "req_sg" ]; then
    python deploy.py --pipeline=req_sg  --model_name=mtl_seq_esmm_v4  --edp_version=20250224 --region=in --debug_v='' --debug=log --ds=20250224
elif [ $1 = "req_sg1" ]; then
    python deploy.py --pipeline=req_sg  --model_name=mtl_seq_esmm_v4  --edp_version=20250224 --region=in --debug_v='' --debug=1 --ds=20250224
fi
