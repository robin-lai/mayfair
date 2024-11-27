# encoding:utf-8

import gc, time,os, json
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

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
BUCKET = 'warehouse-algo'
BUCKET_PREFIX = 's3://%s/' % BUCKET
REC_DIR = 'rec/'

site = 'in'
model = 'all_ctr_din_seq_off_v0'
tfr_sample_dir = 'rec/cn_rec_detail_sample_v1_tfr-all/'
cur_model_root_dir = REC_DIR + model + '/'
eval_pred_dir = cur_model_root_dir + 'eval/'
code_dir      = cur_model_root_dir + 'code/'
code_dir_s3 = BUCKET_PREFIX + code_dir
job_dir       = cur_model_root_dir + 'job/'
job_dir_s3 = BUCKET_PREFIX + job_dir
model_dir     = cur_model_root_dir + 'model/'
deploy_sites = ['sg'] # [site] if site != 'all' else 'sg in'.split()
# train_days = 40
# train_range = ['20241111', '20241112']
# target_date = train_range[-1]

sg_estimator = TensorFlow(
        entry_point='edp_eval.py',
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
        framework_version="1.15",
        py_version='py37',
        max_run=3600*24*3,
        keep_alive_period_in_seconds=1800,
        hyperparameters={
            "mode": "train",
            "epochs":1,
            "linear_lr": 0.005 * 9,
            "dnn_lr": 0.01 * 9,
            "batch_size": 2048,
            # "batch_norm": False,
            "hidden_units": "256,64,32",
            "dnn_dropout": 0,
            "target": 'ctr' if 'ctr' in model else 'cvr',
        },
        metric_definitions=[
           {'Name': 'auc:', 'Regex': 'auc=(.*?);'},
           {'Name': 'loss:', 'Regex': 'loss=(.*?);'},
        ],
        tags=[{'Key': 'cost-team', 'Value': 'algorithm'}],
    )


def ts2date(ts, fmt='%Y%m%d', offset=3600 * 8):
    return time.strftime(fmt, time.localtime(ts + offset))
# Train params:  {'inputs': {'train': 's3://warehouse-algo/rec/cn_rec_detail_sample_v0_tfr-all/ds=20241101', 'eval': 's3://warehouse-algo/rec/cn_rec_detail_sample_v0_tfr-all/ds=20241102'}, 'job_name': 'Job-search-in-ctr-dnn-v0-11-08-17-15-30-1'}
# train_params = {'inputs': {'train': ['s3://warehouse-algo/rec/cn_rec_detail_sample_v0_tfr-all/ds=20241101', 's3://warehouse-algo/rec/cn_rec_detail_sample_v0_tfr-all/ds=20241102'],
#                                'eval': ['s3://warehouse-algo/rec/cn_rec_detail_sample_v0_tfr-all/ds=20241102'],
#                               },
#                     'job_name': 'Job-%s' % (model_name)}

train_params = {'inputs': {'train': 's3://warehouse-algo/rec/cn_rec_detail_sample_v1_tfr-all/ds=20241111',
                               'eval': 's3://warehouse-algo/rec/cn_rec_detail_sample_v1_tfr-all/ds=20241112',
                              },
                    'job_name': 'Job-laidehe-test-%s-%s' % (model.replace('_', '-'),ts2date(time.time(), '%m-%d-%H-%M-%S'))}
print('Train params: ', train_params)
sg_estimator.fit(**train_params)
del sg_estimator
gc.collect()
# alert(ctx)







