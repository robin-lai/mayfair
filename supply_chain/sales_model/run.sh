if [ $1 = "in_c" ]; then
    python -u in_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=7  > log/in_c_519.log 2>&1 &
    sleep 60
    python -u in_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=6  > log/in_c_519.log 2>&1 &
    sleep 60
    python -u in_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=5  > log/in_c_519.log 2>&1 &
    sleep 60
    python -u in_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=4  > log/in_c_519.log 2>&1 &
    sleep 60
    python -u in_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=3  > log/in_c_519.log 2>&1 &
    sleep 60
    python -u in_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=2  > log/in_c_519.log 2>&1 &
    sleep 60
    python -u in_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=1  > log/in_c_519.log 2>&1 &
elif [ $1 = "in_no_c" ]; then
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=7  > log/in_c_no_519.log 2>&1 &
    sleep 60
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=6  > log/in_c_no_519.log 2>&1 &
    sleep 60
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=5  > log/in_c_no_519.log 2>&1 &
    sleep 60
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=4  > log/in_c_no_519.log 2>&1 &
    sleep 60
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=3  > log/in_c_no_519.log 2>&1 &
    sleep 60
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=2  > log/in_c_no_519.log 2>&1 &
    sleep 60
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred,eval --time_delta=1  > log/in_c_no_519.log 2>&1 &
elif [ $1 = "iq_c" ]; then
    python gen_tfrecored_multi_process_with_gen_mt_sample_select.py --ds=$2
fi
