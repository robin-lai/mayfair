# encoding:utf-8
import argparse
import gc, datetime, time
import os

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from aws_auth_init import *

import sys
from pathlib import Path

import requests
def alert_feishu(msg, at_all=True):
    at_str = '<at user_id="all"></at>' if at_all else ''
    x = requests.post('https://open.feishu.cn/open-apis/bot/v2/hook/bb04f4f5-cc78-495b-8b59-c91b669dd55b',
                     headers={'Content-Type': 'application/json'},
                     json={'msg_type': 'text', 'content': {'text': at_str + str(msg)}},
    )
    print('Alert by Feishu', x)

# print(sys.path)
# sys.path.append(str(Path(__file__).absolute().parent.parent.parent.parent))
# sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
print(sys.path)
# from algo_rec.utils.util import add_job_monitor, alert_feishu

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
    job_name = 'lai%s%s' % (args.model_name.replace('_', ''), ts2date(time.time(), '%m-%d-%H-%M'))
    code_dir_s3 = 's3://warehouse-algo/rec/%s/%s/ds=%s/code/' % (args.model_dir, args.model_name, args.train_ds)
    model_dir_s3_prefix = 's3://warehouse-algo/rec/%s/%s/ds=%s/' % (args.model_dir, args.model_name, args.train_ds)
    model_dir_s3 = model_dir_s3_prefix + job_name + '/model/'
    model_dir_s3_pre = 's3://warehouse-algo/rec/%s/%s/ds=%s/model/' % (args.model_dir, args.model_name, args.pre_ds)
    print('model_dir_s3_pre:', model_dir_s3_pre)
    print('model_dir_s3:', model_dir_s3)
    hps = {
        "mode": args.mode,
        "hidden_units": "1024,256,128,64,32",
        "task": args.task,
        "version": "seq_mask_on",
        "initialize": args.initialize
    }
    if args.warm_start_from == 'NEWEST':
        hps['warm_start_from'] = model_dir_s3_pre
    if args.mode == 'infer':
        hps['pred_local'] = args.model_name + '_' + args.eval_ds + '.pkl'
        hps['pred_s3'] = 's3://warehouse-algo/rec/model_pred/%s_%s.pkl' % (args.model_name, args.eval_ds)
    if args.site_code is not None:
        hps['site_code'] = args.site_code
        print('set site_code to:', args.site_code)

    sg_estimator = TensorFlow(
        entry_point='run_rec_model.py',
        dependencies=['aws_auth_init.py','build_feature_columns.py','feature_serv_describe.py'
            ,'din_mask_esmm.py','attention.py','din_base.py', 'din_mask_ctr.py', 'din_mask_esmm.py'],
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
        max_run=3600,
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
            'train': 's3://warehouse-algo/rec/%s/ds=%s' % (args.sample, args.train_ds),
            'eval': 's3://warehouse-algo/rec/%s/ds=%s' % (args.sample, args.eval_ds)
        },
        'job_name': job_name
    }
    print('Train params: ', train_params)
    sg_estimator.fit(**train_params)
    if args.mode == 'train':
        os.system('aws s3 cp --recursive %s %s' % (
        model_dir_s3, model_dir_s3_prefix + 'model'))  # cp can create dest dir,// is wrong
    del sg_estimator
    gc.collect()
    # alert(ctx)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(prog='run_rec_model_di',
                                    description='run_rec_model_di',
                                    epilog='run_rec_model_di')
    today = datetime.date.today().strftime('%Y%m%d')
    parse.add_argument('--task', type=str, default='mtl')
    parse.add_argument('--mode', type=str, default='train')
    parse.add_argument('--sample', type=str, default="cn_rec_detail_sample_v30_savana_in_tfr_row_n300_sample_select")
    parse.add_argument('--site_code', type=str, default=None)
    parse.add_argument('--range', type=str, default='')
    parse.add_argument('--train_ds', type=str,
                       default=(datetime.date.today() - datetime.timedelta(days=2)).strftime('%Y%m%d'))
    parse.add_argument('--eval_ds', type=str, default='20250119eval')
    parse.add_argument('--pre_ds', type=str,
                       default=(datetime.date.today() - datetime.timedelta(days=3)).strftime('%Y%m%d'))
    # parse.add_argument('--model_name', type=str, default='prod-ctr-seq-off-din-v0-test')
    parse.add_argument('--model_name', type=str, default='mtl_seq_esmm_v5')
    parse.add_argument('--model_dir', type=str, default='prod_model')
    parse.add_argument('--warm_start_from', type=str, default='NEWEST')
    parse.add_argument('--initialize', type=str, default='zero')
    parse.add_argument('--instance_count', type=int, default=3)
    args = parse.parse_args()
    if args.range != '':
        ds_range = args.range.split(',')
        for i in range(1, len(ds_range)):
            args.train_ds = ds_range[i]
            args.pre_ds = ds_range[i - 1]
            print('train ds:', args.train_ds)
            st = time.time()
            main(args)
            print('end train ds:%s cost:%s' % (args.train_ds, str(time.time() - st)))
            alert_feishu(f"train ds:{args.train_ds} complete, cost:{str(time.time() - st)}, please check model")
    else:
        print('eval ds:', args.eval_ds)
        st = time.time()
        main(args)
        ed = time.time()
        # job_d = {"start_time": str(st), "end_time": str(ed), "cost": str(ed - st)}
        # add_job_monitor('train', job_d)
        print('end train ds:%s cost:%s' % (args.eval_ds, str(time.time() - st)))
        alert_feishu(f"train ds:{args.train_ds} complete, cost:{str(time.time() - st)}, please check model")
