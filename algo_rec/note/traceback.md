```text
ValueError: The argument 'country' (value Tensor("country:0", shape=(1,), dtype=string)) is not compatible with the shape this function was traced with. Expected shape (?, 1), but got shape (1,).
```
```txt
Traceback (most recent call last):
  File "entry_point_dev.py", line 310, in <module>
    main(FLAGS)
  File "entry_point_dev.py", line 258, in main
    estimator.train(input_fn=train_input_fn, max_steps=None)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 370, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1161, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1188, in _train_model_default
    input_fn, ModeKeys.TRAIN))
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1025, in _get_features_and_labels_from_input_fn
    self._call_input_fn(input_fn, mode))
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1116, in _call_input_fn
    return input_fn(**kwargs)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/inputs/numpy_io.py", line 177, in input_fn
    if len(set(v.shape[0] for v in ordered_dict_data.values())) != 1:
TypeError: unhashable type: 'Dimension'
```

```txt
Traceback (most recent call last):
  File "entry_point_dev.py", line 307, in <module>
    main(FLAGS)
  File "entry_point_dev.py", line 255, in main
    estimator.train(input_fn=train_input_fn, max_steps=None)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 370, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1161, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1188, in _train_model_default
    input_fn, ModeKeys.TRAIN))
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1025, in _get_features_and_labels_from_input_fn
    self._call_input_fn(input_fn, mode))
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1116, in _call_input_fn
    return input_fn(**kwargs)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/inputs/numpy_io.py", line 177, in input_fn
    if len(set(v.shape[0] for v in ordered_dict_data.values())) != 1:
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/inputs/numpy_io.py", line 177, in <genexpr>
    if len(set(v.shape[0] for v in ordered_dict_data.values())) != 1:
AttributeError: 'list' object has no attribute 'shape'
```

```txt
Traceback (most recent call last):
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1365, in _do_call
    return fn(*args)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1350, in _run_fn
    target_list, run_metadata)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1443, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.InternalError: Unsupported object type float

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "entry_point_dev.py", line 295, in <module>
    main(FLAGS)
  File "entry_point_dev.py", line 243, in main
    estimator.train(input_fn=train_input_fn, max_steps=None)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 370, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1161, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1195, in _train_model_default
    saving_listeners)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1495, in _train_with_estimator_spec
    any_step_done = True
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 861, in __exit__
    self._close_internal(exception_type)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 899, in _close_internal
    self._sess.close()
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 1166, in close
    self._sess.close()
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 1334, in close
    ignore_live_threads=True)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/training/coordinator.py", line 389, in join
    six.reraise(*self._exc_info_to_raise)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/six.py", line 718, in reraise
    raise value.with_traceback(tb)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/inputs/queues/feeding_queue_runner.py", line 94, in _run
    sess.run(enqueue_op, feed_dict=feed_dict)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 956, in run
    run_metadata_ptr)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1180, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1359, in _do_run
    run_metadata)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1384, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InternalError: Unsupported object type float
```

```txt
Traceback (most recent call last):
  File "entry_point_dev.py", line 307, in <module>
    main(FLAGS)
  File "entry_point_dev.py", line 267, in main
    print("predictions:", predictions.eval())
AttributeError: 'generator' object has no attribute 'eval'
```

```txt
Traceback (most recent call last):
  File "entry_point_dev.py", line 309, in <module>
    main(FLAGS)
  File "entry_point_dev.py", line 286, in main
    estimator.export_savedmodel(args.model_dir, make_serving_input_receiver_fn())
  File "entry_point_dev.py", line 274, in make_serving_input_receiver_fn
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns_new)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/feature_column/feature_column.py", line 806, in make_parse_example_spec
    'Given: {}'.format(column))
ValueError: All feature_columns must be _FeatureColumn instances. Given: goodsid_fc_emb_list
```

```txt
Traceback (most recent call last):
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1365, in _do_call
    return fn(*args)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1350, in _run_fn
    target_list, run_metadata)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1443, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Number of batches of 'then' must match size of 'cond', but saw: 100 vs. 5
         [[{{node attention_goodsid/Select}}]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "entry_point_dev.py", line 346, in <module>
    main(FLAGS)
  File "entry_point_dev.py", line 285, in main
    estimator.train(input_fn=train_input_fn, max_steps=None)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 370, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1161, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1195, in _train_model_default
    saving_listeners)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1494, in _train_with_estimator_spec
    _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 754, in run
    run_metadata=run_metadata)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 1259, in run
    run_metadata=run_metadata)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 1360, in run
    raise six.reraise(*original_exc_info)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/six.py", line 719, in reraise
    raise value
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 1345, in run
    return self._sess.run(*args, **kwargs)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 1418, in run
    run_metadata=run_metadata)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 1176, in run
    return self._sess.run(*args, **kwargs)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 956, in run
    run_metadata_ptr)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1180, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1359, in _do_run
    run_metadata)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1384, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Number of batches of 'then' must match size of 'cond', but saw: 100 vs. 5
         [[node attention_goodsid/Select (defined at /home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'attention_goodsid/Select':
  File "entry_point_dev.py", line 346, in <module>
    main(FLAGS)
  File "entry_point_dev.py", line 285, in main
    estimator.train(input_fn=train_input_fn, max_steps=None)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 370, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1161, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1191, in _train_model_default
    features, labels, ModeKeys.TRAIN, self.config)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1149, in _call_model_fn
    model_fn_results = self._model_fn(features=features, **kwargs)
  File "entry_point_dev.py", line 220, in _model_fn
    id_type='goodsid', shape=[40000, 32])
  File "entry_point_dev.py", line 182, in attention_layer
    score_pad = tf.where(mask, score, paddings)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/util/deprecation.py", line 324, in new_func
    return func(*args, **kwargs)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/util/dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/ops/array_ops.py", line 3759, in where
    return gen_math_ops.select(condition=condition, x=x, y=y, name=name)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/ops/gen_math_ops.py", line 9439, in select
    "Select", condition=condition, t=x, e=y, name=name)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
    op_def=op_def)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
    attrs, op_def, compute_device)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
    op_def=op_def)
  File "/home/sagemaker-user/.conda/envs/py37/lib/python3.7/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
    self._traceback = tf_stack.extract_stack()
```

```txt
features: {'country': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:1' shape=(?,) dtype=string>,
 'goods_id': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:2' shape=(?,) dtype=string>,
  'is_clk': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:3' shape=(?,) dtype=int32>, 
  'cate_id': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:4' shape=(?,) dtype=string>, 'ctr_7d': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:5' shape=(?,) dtype=float32>, 'cvr_7d': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:6' shape=(?,) dtype=float32>,
 'seq_goods_id': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:7' shape=(?,) dtype=string>}
```

```txt
features: {'cate_id': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:1' shape=(?,) dtype=int64>,
 'country': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:2' shape=(?,) dtype=string>, 'ctr_7d': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:3' shape=(?,) dtype=float64>, 'cvr_7d': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:4' shape=(?,) dtype=float64>,
 'goods_id': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:5' shape=(?,) dtype=string>,
'seq_goods_id': <tf.Tensor 'random_shuffle_queue_DequeueUpTo:6' shape=(?, 5) dtype=string>}
```


```txt
features_data {'cart_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:0' shape=(1,) dtype=int64>, 'cate_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:1' shape=(1,) dtype=string>, 'cate_level1_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:2' shape=(1,) dtype=string>, 'cate_level2_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:3' shape=(1,) dtype=string>, 'cate_level3_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:4' shape=(1,) dtype=string>, 'cate_level4_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:5' shape=(1,) dtype=string>, 'click_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:6' shape=(1,) dtype=int64>, 'country': <tf.Tensor 'ParseSingleExample/ParseSingleExample:7' shape=(1,) dtype=string>, 'ctr_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:8' shape=(1,) dtype=float32>, 'cvr_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:9' shape=(1,) dtype=float32>, 'goods_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:10' shape=(1,) dtype=string>, 'ord_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:13' shape=(1,) dtype=int64>, 'ord_total': <tf.Tensor 'ParseSingleExample/ParseSingleExample:14' shape=(1,) dtype=int64>, 'pay_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:15' shape=(1,) dtype=int64>, 'pay_total': <tf.Tensor 'ParseSingleExample/ParseSingleExample:16' shape=(1,) dtype=int64>, 'seq_cate_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:17' shape=(?, 20) dtype=string>, 'seq_goods_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:18' shape=(?, 20) dtype=string>, 'show_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:19' shape=(1,) dtype=int64>}
features: {'cart_7d': <tf.Tensor 'IteratorGetNext:0' shape=(?, 1) dtype=int64>, 'cate_id': <tf.Tensor 'IteratorGetNext:1' shape=(?, 1) dtype=string>, 'cate_level1_id': <tf.Tensor 'IteratorGetNext:2' shape=(?, 1) dtype=string>, 'cate_level2_id': <tf.Tensor 'IteratorGetNext:3' shape=(?, 1) dtype=string>, 'cate_level3_id': <tf.Tensor 'IteratorGetNext:4' shape=(?, 1) dtype=string>, 'cate_level4_id': <tf.Tensor 'IteratorGetNext:5' shape=(?, 1) dtype=string>, 'click_7d': <tf.Tensor 'IteratorGetNext:6' shape=(?, 1) dtype=int64>, 'country': <tf.Tensor 'IteratorGetNext:7' shape=(?, 1) dtype=string>, 'ctr_7d': <tf.Tensor 'IteratorGetNext:8' shape=(?, 1) dtype=float32>, 'cvr_7d': <tf.Tensor 'IteratorGetNext:9' shape=(?, 1) dtype=float32>, 'goods_id': <tf.Tensor 'IteratorGetNext:10' shape=(?, 1) dtype=string>, 'ord_7d': <tf.Tensor 'IteratorGetNext:11' shape=(?, 1) dtype=int64>, 'ord_total': <tf.Tensor 'IteratorGetNext:12' shape=(?, 1) dtype=int64>, 'pay_7d': <tf.Tensor 'IteratorGetNext:13' shape=(?, 1) dtype=int64>, 'pay_total': <tf.Tensor 'IteratorGetNext:14' shape=(?, 1) dtype=int64>, 'seq_cate_id': <tf.Tensor 'IteratorGetNext:15' shape=(?, ?, 20) dtype=string>, 'seq_goods_id': <tf.Tensor 'IteratorGetNext:16' shape=(?, ?, 20) dtype=string>, 'show_7d': <tf.Tensor 'IteratorGetNext:17' shape=(?, 1) dtype=int64>}
labels Tensor("IteratorGetNext:18", shape=(?, 1), dtype=int64)
features Tensor("IteratorGetNext:16", shape=(?, ?, 20), dtype=string)
```

```txt
Command "/usr/local/bin/python3.7 entry_point_dev.py --batch_size 1024 --checkpoint s3://warehouse-algo/rec/in_ctr_din_dev_v0/job/ds=20241113/checkpoint --dnn_dropout 0 --dnn_hidden_units 256,64,32 --dnn_lr 0.09 --epochs 1 --linear_lr 0.045 --mode train --model_dir s3://warehouse-algo/rec/in_ctr_din_dev_v0/job/Job-laidehe-test-in-ctr-din-dev-v0-11-18-10-14-32/model --target ctr"
```

```txt
features_data {'cart_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:0' shape=(1,) dtype=int64>, 'cate_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:1' shape=(1,) dtype=string>, 'cate_level1_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:2' shape=(1,) dtype=string>, 'cate_level2_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:3' shape=(1,) dtype=string>, 'cate_level3_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:4' shape=(1,) dtype=string>, 'cate_level4_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:5' shape=(1,) dtype=string>, 'click_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:6' shape=(1,) dtype=int64>, 'country': <tf.Tensor 'ParseSingleExample/ParseSingleExample:7' shape=(1,) dtype=string>, 'ctr_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:8' shape=(1,) dtype=float32>, 'cvr_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:9' shape=(1,) dtype=float32>, 'goods_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:10' shape=(1,) dtype=string>, 'ord_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:13' shape=(1,) dtype=int64>, 'ord_total': <tf.Tensor 'ParseSingleExample/ParseSingleExample:14' shape=(1,) dtype=int64>, 'pay_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:15' shape=(1,) dtype=int64>, 'pay_total': <tf.Tensor 'ParseSingleExample/ParseSingleExample:16' shape=(1,) dtype=int64>, 'seq_cate_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:17' shape=(?, 20) dtype=string>, 
'seq_goods_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:18' shape=(?, 20) dtype=string>, 'show_7d': <tf.Tensor 'ParseSingleExample/ParseSingleExample:19' shape=(1,) dtype=int64>}
features: {'cart_7d': <tf.Tensor 'IteratorGetNext:0' shape=(?, 1) dtype=int64>, 'cate_id': <tf.Tensor 'IteratorGetNext:1' shape=(?, 1) dtype=string>, 'cate_level1_id': <tf.Tensor 'IteratorGetNext:2' shape=(?, 1) dtype=string>, 'cate_level2_id': <tf.Tensor 'IteratorGetNext:3' shape=(?, 1) dtype=string>, 'cate_level3_id': <tf.Tensor 'IteratorGetNext:4' shape=(?, 1) dtype=string>, 'cate_level4_id': <tf.Tensor 'IteratorGetNext:5' shape=(?, 1) dtype=string>, 'click_7d': <tf.Tensor 'IteratorGetNext:6' shape=(?, 1) dtype=int64>, 'country': <tf.Tensor 'IteratorGetNext:7' shape=(?, 1) dtype=string>, 'ctr_7d': <tf.Tensor 'IteratorGetNext:8' shape=(?, 1) dtype=float32>, 'cvr_7d': <tf.Tensor 'IteratorGetNext:9' shape=(?, 1) dtype=float32>, 'goods_id': <tf.Tensor 'IteratorGetNext:10' shape=(?, 1) dtype=string>, 'ord_7d': <tf.Tensor 'IteratorGetNext:11' shape=(?, 1) dtype=int64>, 'ord_total': <tf.Tensor 'IteratorGetNext:12' shape=(?, 1) dtype=int64>, 'pay_7d': <tf.Tensor 'IteratorGetNext:13' shape=(?, 1) dtype=int64>, 'pay_total': <tf.Tensor 'IteratorGetNext:14' shape=(?, 1) dtype=int64>, 'seq_cate_id': <tf.Tensor 'IteratorGetNext:15' shape=(?, ?, 20) dtype=string>, 
'seq_goods_id': <tf.Tensor 'IteratorGetNext:16' shape=(?, ?, 20) dtype=string>, 'show_7d': <tf.Tensor 'IteratorGetNext:17' shape=(?, 1) dtype=int64>}
labels Tensor("IteratorGetNext:18", shape=(?, 1), dtype=int64)
features Tensor("IteratorGetNext:16", shape=(?, ?, 20), dtype=string)
raw seq_ipt tensor shape: (?, ?, 20)
raw tid_ipt tensor shape: (?, 1)
seq_emb (?, ?, 20, 32)
tid_emb (?, 1, 1, 32)
```

```txt
Traceback (most recent call last):
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/framework/tensor_util.py", line 541, in make_tensor_proto
    str_values = [compat.as_bytes(x) for x in proto_values]
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/framework/tensor_util.py", line 541, in <listcomp>
    str_values = [compat.as_bytes(x) for x in proto_values]
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/util/compat.py", line 71, in as_bytes
    (bytes_or_text,))
TypeError: Expected binary or unicode string, got 1
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "entry_point_dev.py", line 367, in <module>
    main(FLAGS)
2024-11-18 02:20:19,103 sagemaker-training-toolkit ERROR    ExecuteUserScriptError:
  File "entry_point_dev.py", line 306, in main
    estimator.train(input_fn=train_input_fn, max_steps=None)
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/util/smdebug.py", line 57, in run
    return_value = function(*args, **kwargs)
  File "/usr/local/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 370, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/usr/local/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1161, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/usr/local/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1191, in _train_model_default
    features, labels, ModeKeys.TRAIN, self.config)
  File "/usr/local/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1149, in _call_model_fn
    model_fn_results = self._model_fn(features=features, **kwargs)
  File "entry_point_dev.py", line 241, in _model_fn
    id_type='goodsid', shape=[40000, 32])
  File "entry_point_dev.py", line 193, in attention_layer
    tid_emb_tile = tf.tile(tid_emb, [1, seq_len, 1])
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/ops/gen_array_ops.py", line 11309, in tile
    "Tile", input=input, multiples=multiples, name=name)
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/framework/op_def_library.py", line 531, in _apply_op_helper
    raise err
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/framework/op_def_library.py", line 528, in _apply_op_helper
    preferred_dtype=default_dtype)
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/framework/ops.py", line 1297, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/framework/constant_op.py", line 286, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/framework/constant_op.py", line 227, in constant
    allow_broadcast=True)
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/framework/constant_op.py", line 265, in _constant_impl
    allow_broadcast=allow_broadcast))
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/framework/tensor_util.py", line 545, in make_tensor_proto
    "supported type." % (type(values), values))
TypeError: Failed to convert object of type <class 'list'> to Tensor. Contents: [1, Dimension(None), 1]. Consider casting elements to a supported type.
Command "/usr/local/bin/python3.7 entry_point_dev.py --batch_size 1024 --checkpoint s3://warehouse-algo/rec/in_ctr_din_dev_v0/job/ds=20241113/checkpoint --dnn_dropout 0 --dnn_lr 0.09 --epochs 1 --hidden_units 256,64,32 --linear_lr 0.045 --mode train --model_dir s3://warehouse-algo/rec/in_ctr_din_dev_v0/job/Job-laidehe-test-in-ctr-din-dev-v0-11-18-10-19-57/model --target ctr"
```

```txt
'seq_goods_id': [''],  feature {
    key: "seq_goods_id"
    value {
      bytes_list {
        value: ","
      }
    }
  }
  
  
```

