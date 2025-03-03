#python deploy.py --pipeline=pkg  --model_name=mtl_seq_esmm_v2  --edp_version=20250101-20250119
#python deploy.py --pipeline=pkg  --model_name=mtl_seq_esmm_v2  --edp_version=20250120-20250126
#python deploy.py --pipeline=pkg  --model_name=mtl_seq_esmm_v2  --edp_version=20250302

if [ $1 = "pkg" ]; then
    python deploy.py --pipeline=pkg  --model_name=mtl_seq_esmm_v2  --edp_version=20250302 --region=in 
elif [ $1 = "edp" ]; then
    python deploy.py --pipeline=edp  --model_name=mtl_seq_esmm_v2  --edp_version=20250302 --region=sg --instance_count=1
elif [ $1 = "update" ]; then
    python deploy.py --pipeline=update  --model_name=mtl_seq_esmm_v2  --edp_version=20250302 --region=in 
elif [ $1 = "local" ]; then # change base_data_dir on inference.py
    python inference.py
elif [ $1 = "req_sg" ]; then
    python deploy.py --pipeline=req_sg  --model_name=mtl_seq_esmm_v2  --edp_version=20250302 --region=sg
    python deploy.py --pipeline=req_sg  --model_name=mtl_seq_esmm_v2  --edp_version=20250302 --region=in
elif [ $1 = "req_sg_log" ]; then
    python deploy.py --pipeline=req_sg  --model_name=mtl_seq_esmm_v2  --edp_version=20250302 --region=sg  --debug=log
    python deploy.py --pipeline=req_sg  --model_name=mtl_seq_esmm_v2  --edp_version=20250302 --region=in --debug=log
elif [ $1 = "req_sg1" ]; then
    python deploy.py --pipeline=req_sg  --model_name=mtl_seq_esmm_v2  --edp_version=20250302 --region=in  --debug=1
fi

