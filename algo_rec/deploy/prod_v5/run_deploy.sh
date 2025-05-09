if [ $1 = "pkg" ]; then
    python deploy.py --pipeline=pkg  --model_name=mtl_seq_esmm_v5  --edp_version=$2 --region=in --debug_v='' --ds=$2
elif [ $1 = "edp" ]; then
    python deploy.py --pipeline=edp  --model_name=mtl_seq_esmm_v5  --edp_version=$2 --region=sg --debug_v='' --ds=$2 --instance_count=1
elif [ $1 = "edp_in" ]; then
    python deploy.py --pipeline=edp  --model_name=mtl_seq_esmm_v5  --edp_version=$2 --region=in --debug_v='' --ds=$2 --instance_count=1
elif [ $1 = "update" ]; then
    python deploy.py --pipeline=update  --model_name=mtl_seq_esmm_v5  --edp_version=$2 --region=in --debug_v='' --ds=$2
elif [ $1 = "local" ]; then # change base_data_dir on inference.py
    python inference.py
elif [ $1 = "req_sg" ]; then
    python deploy.py --pipeline=req_sg  --model_name=mtl_seq_esmm_v5  --edp_version=$2 --region=sg --debug_v='' --debug=log --ds=$2
    python deploy.py --pipeline=req_sg  --model_name=mtl_seq_esmm_v5  --edp_version=$2 --region=in --debug_v='' --debug=log --ds=$2
elif [ $1 = "req_sg1" ]; then
    python deploy.py --pipeline=req_sg  --model_name=mtl_seq_esmm_v5  --edp_version=$2 --region=in --debug_v='' --debug=1 --ds=$2
fi
