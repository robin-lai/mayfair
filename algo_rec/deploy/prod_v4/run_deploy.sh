if [ $1 = "pkg" ]; then
    python deploy.py --pipeline=pkg  --model_name=mtl_seq_esmm_v4  --edp_version=20250129-20250204
elif [ $1 = "edp" ]; then
    python deploy.py --pipeline=pkg  --model_name=mtl_seq_esmm_v4  --edp_version=20250129-20250204
fi
