if [ $1 = "recall" ]; then
    python -u swing_multi_process.py --pre_ds=20250219
elif [ $1 = "recall2" ]; then
    python -u swing_multi_process.py --pre_ds=20250206,20250207,20250208,20250209,20250210
elif [ $1 = "recall_d" ]; then
    python -u swing_multi_process.py --pre_ds=20250119 --sample_num=10 --p=1
elif [ $1 = "bi" ]; then
    python -u swing_result2bi.py --ds=20250216
elif [ $1 = "u2i2i" ]; then
    python -u recall_u2i2i.py --pre_ds=20250224
fi
# nohup python -u swing_multi_process.py --pre_ds=20250106 --beta=0.6 done
# nohup python -u swing_multi_process.py --pre_ds=20250106 --beta=0.7 done
# nohup python -u swing_multi_process.py --pre_ds=20250107 --beta=0.7 --ubeta=0.5 --alph=1.0 --p=10  cpu vs mem: 6:8 r5.4x  6440s
# nohup python -u swing_multi_process.py --pre_ds=20250108,20250109,20250110,20250111,20250112,20250113,20250114,20250115,20250116 --beta=0.7 --ubeta=0.5 --alph=1.0 --p=10 done
# nohup python -u swing_multi_process.py --pre_ds=20241228,20241229,20241230,20241231,20250101,20250102,20250103,20250104,20250105 --beta=0.7 --ubeta=0.5 --alph=1.0 --p=8 run
# nohup python -u swing_multi_process.py --pre_ds=20241217,20241218,20241219,20241220,20241221,20241222,20241223 --beta=0.7 --ubeta=0.5 --alph=1.0 --p=10 done
# nohup python -u swing_multi_process.py --pre_ds=20241222,20241223,20241224,20241225,20241226,20241227 --beta=0.7 --ubeta=0.5 --alph=1.0 --p=8 wait
# error ds 20241228
# 10内存会满
# nohup python -u swing_multi_process.py --pre_ds=20250117,20250118,20250119 --beta=0.7 --ubeta=0.5 --alph=1.0 --p=7
