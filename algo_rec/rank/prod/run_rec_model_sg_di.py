# encoding:utf-8
import argparse
import gc, datetime
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from aws_auth_init import *


def main(args):
    s3_cli = boto3.client('s3')
    sm_cli = boto3.client('sagemaker')
    sm_sess = sagemaker.Session()
    role = get_execution_role()
    print('aws region:', sm_sess.boto_region_name)
    print('role:', role)

    # Basic config
    # job_name = args.model_name + args.ds
    code_dir_s3 = 's3://warehouse-algo/rec/test_model/%s/code/' % args.model_name
    model_dir_s3 = 's3://warehouse-algo/rec/test_model/%s/ds=%s/' % (args.model_name, args.ds)
    model_dir_s3_pre = 's3://warehouse-algo/rec/test_model/%s/ds=%s/' % (args.model_name, args.pre_ds)

    sg_estimator = TensorFlow(
        entry_point='run_rec_model_sg.py',
        dependencies=['aws_auth_init.py'],
        role=role,
        input_mode="Pipe",
        instance_count=args.instance_count,
        instance_type="ml.r5.xlarge",
        distribution={'parameter_server': {'enabled': True}},
        volume_size=250,
        code_location=code_dir_s3,
        output_path=model_dir_s3,
        disable_profiler=True,
        framework_version="1.15.2",
        py_version='py37',
        max_run=3600 * 24 * 3,
        keep_alive_period_in_seconds=1800,
        hyperparameters={
            "mode": "train",
            "hidden_units": "256,64,32",
            "checkpoint": model_dir_s3_pre
        },
        metric_definitions=[
            {'Name': 'auc:', 'Regex': 'auc=(.*?);'},
            {'Name': 'loss:', 'Regex': 'loss=(.*?);'},
        ],
        tags=[{'Key': 'cost-team', 'Value': 'algorithm'}],
    )

    # def ts2date(ts, fmt='%Y%m%d', offset=3600 * 8):
    #     return time.strftime(fmt, time.localtime(ts + offset))

    train_params = {
	    'inputs': {
	    	'train': 's3://warehouse-algo/rec/cn_rec_detail_sample_v1_tfr_ctr/ds=20241111',
	    	'eval': 's3://warehouse-algo/rec/cn_rec_detail_sample_v1_tfr_ctr/ds=20241112'
	    },
	    'job_name': args.job_name
    }
    print('Train params: ', train_params)
    sg_estimator.fit(**train_params)
    del sg_estimator
    gc.collect()
    # alert(ctx)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(prog='run_rec_model_di',
                                    description='run_rec_model_di',
                                    epilog='run_rec_model_di')
    parse.add_argument('--ds', type=str, default=datetime.date.today().strftime('%Y%m%d'))
    parse.add_argument('--pre_ds', type=str,
                       default=(datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y%m%d'))
    # parse.add_argument('--model_name', type=str, default='prod-ctr-seq-off-din-v0-test')
    parse.add_argument('--model_name', type=str, default='prod_ctr_seq_off_din_v0_test')
    parse.add_argument('--instance_count', type=int, default=1)
    parse.add_argument('--job_name', type=str, default='laidehe-rec-job')
    args = parse.parse_args()
    main(args)