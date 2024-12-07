# encoding:utf-8
import argparse
import gc, datetime,time
import os

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from aws_auth_init import *

# steup up
s3_cli = boto3.client('s3')
sm_sess = sagemaker.Session()
print('aws region:', sm_sess.boto_region_name)
sm_cli = boto3.client('sagemaker')
role = get_execution_role()
print('role:', role)

def ts2date(ts, fmt='%Y%m%d', offset=3600 * 8):
    return time.strftime(fmt, time.localtime(ts + offset))

def main(args):
    # Basic config
    job_name = 'Job-laidehe-%s-%s' % (args.model_name.replace('_', '-'), ts2date(time.time(), '%m-%d-%H-%M-%S'))
    code_dir_s3 = 's3://warehouse-algo/rec/%s/%s/ds=%s/code/' % (args.model_dir, args.model_name, args.ds)
    model_dir_s3_prefix = 's3://warehouse-algo/rec/%s/%s/ds=%s/' % (args.model_dir, args.model_name, args.ds)
    model_dir_s3 = model_dir_s3_prefix + job_name + '/model/'
    model_dir_s3_pre = 's3://warehouse-algo/rec/%s/%s/ds=%s/model/' % (args.model_dir, args.model_name, args.pre_ds)
    print('model_dir_s3_pre:', model_dir_s3_pre)
    print('model_dir_s3:', model_dir_s3)
    hps = {
        "mode": args.mode,
        "hidden_units": "256,64,32",
        "task": args.task
    }
    if args.warm_start_from == 'NEWEST':
        hps['warm_start_from'] = model_dir_s3_pre
    if args.mode == 'infer':
        hps['pred_local'] = args.model_name + '_' + args.eval_ds + '.pkl'
        hps['pred_s3'] = 's3://warehouse-algo/rec/model_pred/%s_%s.pkl'%(args.model_name, args.eval_ds)

    sg_estimator = TensorFlow(
        entry_point='run_rec_model.py',
        dependencies=['aws_auth_init.py'],
        role=role,
        input_mode="Pipe",
        instance_count=args.instance_count,
        instance_type="ml.r5.xlarge",
        distribution={'parameter_server': {'enabled': True}},
        volume_size=250,
        code_location=code_dir_s3,
        output_path=model_dir_s3_prefix,
        disable_profiler=True,
        framework_version="1.15.2",
        py_version='py37',
        max_run=3600 * 24 * 3,
        keep_alive_period_in_seconds=1800,
        hyperparameters=hps,
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
	    	'train': 's3://warehouse-algo/rec/%s/ds=%s'%(args.sample, args.train_ds),
	    	'eval': 's3://warehouse-algo/rec/%s/ds=%s'%(args.sample, args.eval_ds)
	    },
	    'job_name': job_name
    }
    print('Train params: ', train_params)
    sg_estimator.fit(**train_params)
    if args.mode == 'train':
        os.system('aws s3 cp --recursive %s %s' % (model_dir_s3, model_dir_s3_prefix + 'model')) # cp can create dest dir,// is wrong
    del sg_estimator
    gc.collect()
    # alert(ctx)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(prog='run_rec_model_di',
                                    description='run_rec_model_di',
                                    epilog='run_rec_model_di')
    today = datetime.date.today().strftime('%Y%m%d')
    parse.add_argument('--task', type=str, default='ctr')
    parse.add_argument('--mode', type=str, default='train')
    parse.add_argument('--sample', type=str, default="cn_rec_detail_sample_v10_ctr")
    parse.add_argument('--ds', type=str, default=today)
    parse.add_argument('--range', type=str, default='')
    parse.add_argument('--train_ds', type=str, default=today)
    parse.add_argument('--eval_ds', type=str, default='20241112')
    parse.add_argument('--pre_ds', type=str,
                       default=(datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y%m%d'))
    # parse.add_argument('--model_name', type=str, default='prod-ctr-seq-off-din-v0-test')
    parse.add_argument('--model_name', type=str, default='prod_ctr_seq_off_din_v0')
    parse.add_argument('--model_dir', type=str, default='prod_model')
    parse.add_argument('--warm_start_from', type=str, default='NEWEST')
    parse.add_argument('--instance_count', type=int, default=1)
    args = parse.parse_args()
    if args.range != '':
        ds_range = args.range.split(',')
        for i in range(1,len(ds_range)):
            args.ds=ds_range[i]
            args.train_ds=ds_range[i]
            args.pre_ds=ds_range[i-1]
            print('train ds:', args.ds)
            st = time.time()
            main(args)
            print('end train ds:%s cost:%s' % (args.ds, str(time.time()-st)))
    else:
        print('eval ds:', args.eval_ds)
        st = time.time()
        main(args)
        print('end train ds:%s cost:%s' % (args.eval_ds, str(time.time() - st)))
