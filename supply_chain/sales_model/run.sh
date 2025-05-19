if [ $1 = "in_c" ]; then
    python -u in_cancel_pipeline.py --pipeline=init,train,pred --time_delta=6
    sleep 60
    python -u in_cancel_pipeline.py --pipeline=init,train,pred --time_delta=5
    sleep 60
    python -u in_cancel_pipeline.py --pipeline=init,train,pred --time_delta=4
    sleep 60
    python -u in_cancel_pipeline.py --pipeline=init,train,pred --time_delta=3
    sleep 60
    python -u in_cancel_pipeline.py --pipeline=init,train,pred --time_delta=2
    sleep 60
    python -u in_cancel_pipeline.py --pipeline=init,train,pred --time_delta=1
elif [ $1 = "in_no_c" ]; then
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred --time_delta=6
    sleep 60
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred --time_delta=5
    sleep 60
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred --time_delta=4
    sleep 60
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred --time_delta=3
    sleep 60
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred --time_delta=2
    sleep 60
    python -u in_no_cancel_pipeline.py --pipeline=init,train,pred --time_delta=1
elif [ $1 = "iq_c" ]; then
    python -u iq_cancel_pipeline.py --pipeline=init,train,pred --time_delta=18
    sleep 60
    python -u iq_cancel_pipeline.py --pipeline=init,train,pred --time_delta=17
    sleep 60
    python -u iq_cancel_pipeline.py --pipeline=init,train,pred --time_delta=16
    sleep 60
    python -u iq_cancel_pipeline.py --pipeline=init,train,pred --time_delta=15
    sleep 60
    python -u iq_cancel_pipeline.py --pipeline=init,train,pred --time_delta=14
    sleep 60
    python -u iq_cancel_pipeline.py --pipeline=init,train,pred --time_delta=13
    sleep 60
    python -u iq_cancel_pipeline.py --pipeline=init,train,pred --time_delta=12
    sleep 60
    python -u iq_cancel_pipeline.py --pipeline=init,train,pred --time_delta=11
    sleep 60
    python -u iq_cancel_pipeline.py --pipeline=init,train,pred --time_delta=10
    sleep 60
fi
