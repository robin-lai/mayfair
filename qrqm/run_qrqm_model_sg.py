# encoding:utf-8

import gc, time,os, json
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from aws_auth_init import *

# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

# steup up

s3_cli = boto3.client('s3')
sm_sess = sagemaker.Session()
print('aws region:', sm_sess.boto_region_name)
sm_cli = boto3.client('sagemaker')
role = get_execution_role()
print('role:', role)

# Basic config
model = 'qrqm_model_test'
code_dir_s3 = 's3://warehouse-algo/rec/%s/code/' % model
job_dir_s3 = 's3://warehouse-algo/rec/%s/job/' % model

sg_estimator = TensorFlow(
        entry_point='run_qrqm_model.py',
        dependencies=['aws_auth_init.py'],
        role=role,
        input_mode="Pipe",
        instance_count=1,
        instance_type="ml.r5.xlarge",
        distribution={'parameter_server': {'enabled': True}},
        volume_size=250,
        code_location= code_dir_s3,
        output_path= job_dir_s3,
        disable_profiler=True,
        framework_version="1.15.2",
        py_version='py37',
        max_run=3600*24*1,
        keep_alive_period_in_seconds=1800,
        hyperparameters={
            "mode": "train",
            "hidden_units": "256,64,32"
        },
        metric_definitions=[
           {'Name': 'auc:', 'Regex': 'auc=(.*?);'},
           {'Name': 'loss:', 'Regex': 'loss=(.*?);'},
        ],
        tags=[{'Key': 'cost-team', 'Value': 'algorithm'}],
    )


def ts2date(ts, fmt='%Y%m%d', offset=3600 * 8):
    return time.strftime(fmt, time.localtime(ts + offset))
train_params = {'inputs': {'train': 's3://algo-sg/qrqm_algo/qrqm_uuid_pgid_bhv_sample_v3_tfr/ds=20241207'
                           ,'eval': 's3://algo-sg/qrqm_algo/qrqm_uuid_pgid_bhv_sample_v3_tfr/ds=20241207'
                              },
                    'job_name': 'Job-laidehe-test-%s-%s' % (model.replace('_', '-'),ts2date(time.time(), '%m-%d-%H-%M-%S'))}
print('Train params: ', train_params)
sg_estimator.fit(**train_params)
del sg_estimator
gc.collect()
# alert(ctx)







