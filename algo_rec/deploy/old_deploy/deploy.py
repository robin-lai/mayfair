# coding: utf-8
import time
import argparse
import json
import os
import pickle
from pyarrow import parquet
import boto3
import sagemaker
from sagemaker import image_uris, get_execution_role
from sagemaker.session import production_variant

todell_dir = '/home/sagemaker-user/todell/tmp'
deploy_dir = '/home/sagemaker-user/mayfair/algo_rec/deploy/'
deploy_pkg_dir = deploy_dir + 'pkg/'
deploy_tmp_dir = deploy_dir + 'tmp/'
deploy_code_dir = deploy_pkg_dir + 'code/'
deploy_data_dir = deploy_tmp_dir + 'data/'

# item features
fts_item_s3_text_dir = 's3://algo-sg/rec/cn_rec_detail_feature_item_base_for_redis/'
fts_item_local_text_dir = deploy_data_dir + 'cn_rec_detail_feature_item_base_for_redis/'
fts_item_pickle = deploy_pkg_dir + 'item_features.pkl'

# fts_user_seq_off_s3 = 's3://algo-sg/rec/cn_rec_detail_feature_user_seq_v2_for_redis/ds=20241201/'
# fts_user_seq_off_pickle = deploy_pkg_dir + 'user_seq_off_features.pkl'

# config
rec_buk = 's3://warehouse-algo/rec/'
in_rec_buk = 's3://algo-rec/rec/model_online/'
sg_rec_buk = 's3://algo-sg/rec/model_online/'
in_s3_tar_file = ""
sg_s3_tar_file = ""

# endpoint = 'ctr-model-debug1121'


def convert_text2pkl(text_dir):
    from os import listdir
    from os.path import isfile, join
    files = [f for f in listdir(text_dir) if isfile(join(text_dir, f))]
    ll = []
    for file in files:
        with open(fts_item_local_text_dir + file, 'r') as fin:
            ll.extend(fin.readlines())
    print('text file lines num:', len(ll))
    m = {}
    # print(ll[0:10])
    for line in ll:
        k, v = line.split(chr(1))
        ll = v.split(chr(2))
        for ele in ll:
            fts_name, fts_value = ele.split(chr(4))
            trim_v = str.rstrip(fts_value, '\n')
            if k in m:
                m[k][fts_name] = trim_v
            else:
                m[k] = {fts_name: trim_v}
    return m


def convert_user_seq2pkl(pt_file):
    pt = parquet.read_table(pt_file)
    m = {}
    for t in zip(pt['uuid'],pt['seq_goods_id'], pt['seq_cate_id']):
        m[t[0]] = {'seq_goods_id': t[1], 'seq_cate_id':t[2]}
    return m

def pkg(args):
    print('init dir')
    code_file = deploy_dir + 'inference.py'
    os.system('rm -rf %s' % deploy_pkg_dir)
    os.system('rm -rf %s' % deploy_tmp_dir)
    os.system('mkdir -p %s' % deploy_pkg_dir)
    os.system('mkdir -p %s' % deploy_tmp_dir)
    os.system('mkdir -p %s' % deploy_code_dir)
    # os.system('mkdir %s' % deploy_data_dir)
    os.system('mkdir %s' % fts_item_local_text_dir)
    os.system('cp %s %s' % (code_file, deploy_code_dir))

    s3_cli = boto3.client('s3')
    sm_sess = sagemaker.Session()
    print('aws region:', sm_sess.boto_region_name)
    sm_cli = boto3.client('sagemaker')
    role = get_execution_role()
    print('role:', role)

    # download files
    s3_model = rec_buk + args.model_dir + args.model_name + args.model_version
    print('s3_model nane:', s3_model)
    os.system('aws s3 cp --recursive %s %s' % (s3_model, deploy_pkg_dir))
    # os.system('cp -r  %s %s' % (model_local, deploy_pkg_dir))
    os.system('aws s3 cp --recursive %s %s' % (fts_item_s3_text_dir, fts_item_local_text_dir))
    item_fts_dict = convert_text2pkl(fts_item_local_text_dir)
    with open(fts_item_pickle, 'wb') as fout:
        pickle.dump(item_fts_dict, fout)

    # user_seq_off_dict = convert_user_seq2pkl(fts_user_seq_off_s3)
    # with open(fts_user_seq_off_pickle, 'wb') as fout:
    #     pickle.dump(user_seq_off_dict, fout)

    # tar
    tar_file = deploy_dir + args.tar_name
    os.system('cd %s ; tar -czvf  %s  %s' % (deploy_pkg_dir, tar_file, './'))
    # upload
    in_s3_tar_file = in_rec_buk + args.tar_name
    print('upload %s to %s' % (tar_file, in_s3_tar_file))
    os.system('aws s3 cp %s %s' % (tar_file, in_s3_tar_file))
    sg_s3_tar_file = sg_rec_buk + args.tar_name
    print('upload %s to %s' % (tar_file, sg_s3_tar_file))
    os.system('aws s3 cp %s %s' % (tar_file, sg_s3_tar_file))

def alert(msg):
    print(msg)
    return
    msg = str(msg)
    max_size = 1000
    if len(msg) > max_size:
        msg = msg[:(max_size - 10)] + ' ...Over'
    sns_cli = boto3.client('sns')
    for phone_number in [13521039521, ]:
        sns_cli.publish(PhoneNumber='+86%s' % phone_number, Message=msg)

def wait_edp_inservice(edp_name, wait_window=3600, wait_interval=10):
    begin = time.time()
    sm_cli = boto3.client('sagemaker')
    while True:
        now = time.time()
        if now - begin > wait_window:
            alert('Endpoint %s DO NOT inservice after %.2f hours, maybe something is wrong' % (
                edp_name, (now - begin) / 3600))
            raise
        edp_status = sm_cli.describe_endpoint(EndpointName=edp_name)
        if edp_status.get('EndpointStatus') != 'InService':
            print('Endpoint %s DO NOT inservice, continue wait after %.2f hours' % (
                edp_name, (now - begin) / 3600))
            time.sleep(wait_interval)
            continue

        print('Endpoint %s inservice now, finished.' % edp_name)
        break

def create_edp(args):
    s3_cli = boto3.client('s3')
    sm_sess = sagemaker.Session()
    print('aws region:', sm_sess.boto_region_name)
    sm_cli = boto3.client('sagemaker')
    role = get_execution_role()
    print('role:', role)
    # def deploy_new_endpoint(model_data,
    #                         endpoint_name,
    #                         instance_type='ml.r5.large',
    #                         instance_count=1,
    #                         retry_times=0):
    # If an endpoint could describe, it exists, and can not be created by deploy.
    instance_type=args.instance_type
    instance_count=1
    retry_times=0
    try:
        print(s3_cli.describe_endpoint(EndpointName=args.endpoint))
        return
    except:
        pass

    # edp_model_name = endpoint_name + '-' + str(random.randint(10000, 19999))
    variant_name = "Variant-xlarge-1"  # start from 1, incr 1 when updating.
    img = sagemaker.image_uris.retrieve(
        framework='tensorflow',
        version='1.15.3',
        region=sm_sess.boto_region_name,
        image_scope='inference',
        instance_type=instance_type
    )

    print('in_s3_tar_file', args.s3_tar_file)
    sm_sess.create_model(
        name=args.endpoint,
        role=role,
        container_defs={
            "Image": img,
            "ModelDataUrl": args.s3_tar_file,
            'Environment': {
                'TF_DISABLE_MKL': '1',
                'TF_DISABLE_POOL_ALLOCATOR': '1',
                # 'SAGEMAKER_SUBMIT_DIRECTORY': '/home/sagemaker-user/mayfair/algo_rec/deploy/tmp/code/',  # Directory inside the container
                # 'SAGEMAKER_PROGRAM': 'inference.py',
            },
        }
    )

    variant1 = production_variant(
        model_name=args.endpoint,
        instance_type=instance_type,
        initial_instance_count=instance_count,
        variant_name=variant_name,
        initial_weight=1,
    )

    sm_sess.endpoint_from_production_variants(
        name=args.endpoint, production_variants=[variant1],
        tags=[{'Key': 'cost-team', 'Value': 'algorithm'}],
    )
    print(sm_cli.describe_endpoint(EndpointName=args.endpoint))
    wait_edp_inservice(args.endpoint)

def request_edp(args):
    inputs_seq = {
        "cate_level1_id": ["1"],
        "cate_level2_id": ["1"],
        "cate_level3_id": ["1"],
        "cate_level4_id": ["1"],
        "country": ["IN"],
        "ctr_7d": [0.1],
        "cvr_7d": [0.1],
        "show_7d": [100],
        "click_7d": [100],
        "cart_7d": [100],
        "ord_total": [100],
        "pay_total": [100],
        "ord_7d": [100],
        "pay_7d": [100],
        "seq_goods_id": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
                         "18",
                         "19", "20"],
        "goods_id": ["1"],
        "seq_cate_id": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
                        "18",
                        "19", "20"],
        "cate_id": ["1"],
    }

    inputs_no_seq = {
        "cate_level1_id": ["1"],
        "cate_level2_id": ["1"],
        "cate_level3_id": ["1"],
        "cate_level4_id": ["1"],
        "country": ["IN"],
        "ctr_7d": [0.1],
        "cvr_7d": [0.1],
        "show_7d": [100],
        "click_7d": [100],
        "cart_7d": [100],
        "ord_total": [100],
        "pay_total": [100],
        "ord_7d": [100],
        "pay_7d": [100]
    }

    ipt4 = {"signature_name": "serving_default", "instances": [inputs_seq, inputs_seq]}
    sg_client = boto3.client("sagemaker-runtime")
    print('inp-json-dump', json.dumps(ipt4))
    # endpoint = 'ctr-model-debug1121'

    res = sg_client.invoke_endpoint(
        EndpointName=args.endpoint,
        Body=json.dumps(ipt4),
        # Body=json.dumps(ipt4).encode('utf-8'),
        # .encode('utf-8'),
        # Body=ipt4,
        ContentType="application/json"
    )
    print(res["Body"].read())
    # res_json = json.loads(res["Body"].read())

def request_sagemaker(args):
    request = {"signature_name": "serving_default", "city": "Menbai", "country": "IN", "debug": "",
                "featureMap": {"userFeatures": {"high_level_seq": ["1327692"] * 20, "low_level_seq": ["1327692"] * 20}},
                "goodsIdList": ["1327692", "1402902"], "ip": "127.0.0.1", "platform": "H5", "province": "Menbai",
                "scene": "detail_rec", "userId": "23221", "userNo": "2321", "uuid": "fxleyu", "version": "8.2.2"}

    req1 = {
  "signature_name": "serving_default",
  "instances": [
    {
      "goods_id": [
        ""
      ],
      "cate_id": [
        "748"
      ],
      "cate_level1_id": [
        "2"
      ],
      "cate_level2_id": [
        "12"
      ],
      "cate_level3_id": [
        "79"
      ],
      "cate_level4_id": [
        "748"
      ],
      "country": [
        ""
      ],
      "show_7d": [
        106063
      ],
      "click_7d": [
        3594
      ],
      "cart_7d": [
        104
      ],
      "ord_total": [
        357
      ],
      "pay_total": [
        314
      ],
      "ord_7d": [
        6
      ],
      "pay_7d": [
        5
      ],
      "ctr_7d": [
        0.0338855208696718
      ],
      "cvr_7d": [
        0.001669449081803005
      ],
      "highLevelSeqListGoods": [
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692"
      ],
      "highLevelSeqListCateId": [
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748"
      ],
      "lowerLevelSeqListGoods": [
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692"
      ],
      "lowerLevelSeqListCateId": [
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748"
      ]
    },
    {
      "goods_id": [
        ""
      ],
      "cate_id": [
        "449"
      ],
      "cate_level1_id": [
        "2"
      ],
      "cate_level2_id": [
        "12"
      ],
      "cate_level3_id": [
        "74"
      ],
      "cate_level4_id": [
        "449"
      ],
      "country": [
        ""
      ],
      "show_7d": [
        84521
      ],
      "click_7d": [
        2124
      ],
      "cart_7d": [
        211
      ],
      "ord_total": [
        272
      ],
      "pay_total": [
        247
      ],
      "ord_7d": [
        29
      ],
      "pay_7d": [
        28
      ],
      "ctr_7d": [
        0.025129849386542988
      ],
      "cvr_7d": [
        0.013653483992467044
      ],
      "highLevelSeqListGoods": [
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692"
      ],
      "highLevelSeqListCateId": [
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748"
      ],
      "lowerLevelSeqListGoods": [
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692",
        "1327692"
      ],
      "lowerLevelSeqListCateId": [
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748",
        "748"
      ]
    }
  ]
}
    request['ipt'] = req1
    request['debug'] = "1"

    sg_client = boto3.client("sagemaker-runtime")

    print('inp-json-dump', json.dumps(request))
    res = sg_client.invoke_endpoint(
        EndpointName=args.endpoint,
        Body=json.dumps(request),
        # Body=json.dumps(ipt4).encode('utf-8'),
        # .encode('utf-8'),
        # Body=ipt4,
        ContentType="application/json"
    )
    ret = json.loads(res["Body"].read())
    result = []
    print('ret score:', ret)
    for i, goods_id in enumerate(request['goodsIdList']):
        # ret['predictions'][i]['goods_id'] = goods_id
        tmp = {
            'goods_id': goods_id,
            'ctr': ret['predictions'][i]['ctr'],
            'cvr': ret['predictions'][i]['cvr'],
            'ctcvr': ret['predictions'][i]['ctcvr']
        }
        result.append(tmp)
    print('final ret:', result)

import numpy as np
def request_sagemaker_time(args):
    request = {"signature_name": "serving_default", "city": "Menbai", "country": "IN", "debug": "",
               "featureMap": {"userFeatures": {"high_level_seq": ["1327692"] * 20, "low_level_seq": ["1327692"] * 20}},
               "goodsIdList": ["1327692", "1402902"], "ip": "127.0.0.1", "platform": "H5", "province": "Menbai",
               "scene": "detail_rec", "userId": "23221", "userNo": "2321", "uuid": "fxleyu", "version": "8.2.2"}
    request['goodsIdList'] = ["1327692"] * args.goods_num
    cost = []
    sg_client = boto3.client("sagemaker-runtime")
    for i in range(args.req_num):
        print('req idx:', i)
        if i % 5 == 0:
            print('sleep 1s')
            time.sleep(10)
        st = time.time()
        res = sg_client.invoke_endpoint(
            EndpointName=args.endpoint,
            Body=json.dumps(request),
            # Body=json.dumps(ipt4).encode('utf-8'),
            # .encode('utf-8'),
            # Body=ipt4,
            ContentType="application/json"
        )
        ed = time.time()
        cost.append(ed - st)
    print('req:', args.n)
    print('goods_id_num:', args.goods_num)
    print('mean cost:', np.mean(cost), 'max cost:', np.max(cost), 'min cost:', np.min(cost))


def main(args):
    print('before pkg, need inference.py')
    print('start run pipeline:', args.pipeline)
    if 'pkg' in args.pipeline:
        print('start pkg')
        pkg(args)
        print('end pkg')
    if 'edp' in args.pipeline:
        print('start edp')
        if args.region == 'in':
            args.s3_tar_file = in_rec_buk + args.tar_name
        if args.region == 'sg':
            args.s3_tar_file = sg_rec_buk + args.tar_name
        create_edp(args)
        print('end edp')
    if 'req_edp' in args.pipeline:
        print('start request edp')
        request_edp(args)
        print('end request edp')
    if 'req_sg' in args.pipeline:
        print('start request sagemaker')
        request_sagemaker(args)
        print('end request sagemaker')
    if 'time' in args.pipeline:
        request_sagemaker_time(args)

    print('end run pipeline:', args.pipeline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='deploy',
        description='deploy',
        epilog='deploy')
    parser.add_argument('--pipeline', default='pkg,edp,req_sg')
    parser.add_argument('--endpoint', default='test-edp-model')
    parser.add_argument('--region', default='sg')
    parser.add_argument('--edp_version', default='v0')
    parser.add_argument('--model_dir', default='prod_model/')
    parser.add_argument('--model_name', default='prod_mtl_seq_on_esmm_v0')
    parser.add_argument('--model_version', default='/ds=20241203/model/')
    parser.add_argument('--tar_name', default='prod_mtl_seq_on_esmm_v0_v1.tar.gz')
    parser.add_argument('--instance_type', default='ml.r5.large')
    parser.add_argument('--req_num', type=int,  default=10000)
    parser.add_argument('--goods_num', type=int,  default=100)

    args = parser.parse_args()
    args.endpoint = 'edp-' + args.model_name.replace('_', '-') + '-' + args.edp_version
    print('endpoint:', args.endpoint)
    main(args)


