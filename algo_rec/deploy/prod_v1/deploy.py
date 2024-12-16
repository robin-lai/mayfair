# coding: utf-8
import boto3
import sagemaker
from numba.cpython.listobj import in_seq
from sagemaker import get_execution_role
# import sh
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
deploy_dir = '/home/sagemaker-user/mayfair/algo_rec/deploy/prod_v1/'
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
        ll = v.split(chr(4))
        for ele in ll:
            fts_name, fts_value = ele.split(chr(2))
            trim_v = str.rstrip(fts_value, '\n')
            if k in m:
                m[k][fts_name] = trim_v
            else:
                m[k] = {fts_name: trim_v}
    return m


def convert_user_seq2pkl(pt_file):
    pt = parquet.read_table(pt_file)
    m = {}
    for t in zip(pt['uuid'], pt['seq_goods_id'], pt['seq_cate_id']):
        m[t[0]] = {'seq_goods_id': t[1], 'seq_cate_id': t[2]}
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
    instance_type = args.instance_type
    instance_count = 1
    retry_times = 0
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


def request_sagemaker(args):
    request = {
        "signature_name": "serving_default",
        "city": "Menbai",
        "country": "IN",
        "debug": "",
        "parentGoodsId": "1327692",
        "featureMap": {
            "userFeatures": {
                "high_level_seq": [
                    "1327692",
                    "1402902",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    ""
                ],
                "low_level_seq": [
                    "1327692",
                    "1402902",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    ""
                ],
                "user_feature_context": {
                    "register_brand": "other",
                    "last_login_device": "huawei",
                    "last_login_brand": "huawei"
                }
            }
        },
        "goodsIdList": [
            "1327692",
            "1402902"
        ],
        "ip": "127.0.0.1",
        "platform": "H5",
        "province": "Menbai",
        "scene": "detail_rec",
        "userId": "23221",
        "userNo": "2321",
        "uuid": "fxleyu",
        "version": "8.2.2"
    }

    req_row = {
        "signature_name": "serving_default",
        "instances": [
            {
                "highLevelSeqListGoods": [
                    "1327692",
                    "1402902",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    ""
                ],
                "highLevelSeqListCateId": [
                    "748",
                    "449",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    ""
                ],
                "lowerLevelSeqListGoods": [
                    "1327692",
                    "1402902",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    ""
                ],
                "lowerLevelSeqListCateId": [
                    "748",
                    "449",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    ""
                ],
                "register_brand": ["other"],
                "last_login_device": ["huawei"],
                "last_login_brand": ["huawei"],
                "main_goods_id": ["1402902"],
                "main_cate_id": [
                    "449"
                ],
                "main_cate_level2_id": [
                    "12"
                ],
                "main_cate_level3_id": [
                    "74"
                ],
                "main_cate_level4_id": [
                    "449"
                ],
                "goods_id": [
                    "1327692"
                ],
                "cate_id": [
                    "748"
                ],
                "is_rel_cate": [
                    0
                ],
                "cate_level1_id": [
                    "2"
                ],
                "cate_level2_id": [
                    "12"
                ],
                "is_rel_cate2": [
                    1
                ],
                "cate_level3_id": [
                    "79"
                ],
                "is_rel_cate3": [
                    0
                ],
                "cate_level4_id": [
                    "748"
                ],
                "is_rel_cate4": [
                    0
                ],
                "country": [
                    ""
                ],
                "prop_seaon": [
                    "Summer"
                ],
                "prop_length": [
                    "Maxi"
                ],
                "prop_main_material": [
                    "Polyester"
                ],
                "prop_pattern": [
                    "Solid"
                ],
                "prop_style": [
                    "Sexy | Extravagant | Elegant"
                ],
                "prop_quantity": [
                    "1"
                ],
                "prop_fitness": [
                    "Regular Fit"
                ],
                "show_7d": [
                    56152
                ],
                "click_7d": [
                    1960
                ],
                "cart_7d": [
                    51
                ],
                "ord_total": [
                    363
                ],
                "pay_total": [
                    320
                ],
                "ord_7d": [
                    5
                ],
                "pay_7d": [
                    5
                ],
                "sales_price": [
                    0
                ],
                "ctr_7d": [
                    0.03490525715913948
                ],
                "cvr_7d": [
                    0.002551020408163265
                ]
            },
            {
                "highLevelSeqListGoods": [
                    "1327692",
                    "1402902",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    ""
                ],
                "highLevelSeqListCateId": [
                    "748",
                    "449",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    ""
                ],
                "lowerLevelSeqListGoods": [
                    "1327692",
                    "1402902",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    ""
                ],
                "lowerLevelSeqListCateId": [
                    "748",
                    "449",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    ""
                ],
                "register_brand": ["other"],
                "last_login_device": ["huawei"],
                "last_login_brand": ["huawei"],
                "main_goods_id": ["1402902"],
                "main_cate_id": [
                    "449"
                ],
                "main_cate_level2_id": [
                    "12"
                ],
                "main_cate_level3_id": [
                    "74"
                ],
                "main_cate_level4_id": [
                    "449"
                ],
                "goods_id": [
                    "1327692"
                ],
                "cate_id": [
                    "748"
                ],
                "is_rel_cate": [
                    0
                ],
                "cate_level1_id": [
                    "2"
                ],
                "cate_level2_id": [
                    "12"
                ],
                "is_rel_cate2": [
                    1
                ],
                "cate_level3_id": [
                    "79"
                ],
                "is_rel_cate3": [
                    0
                ],
                "cate_level4_id": [
                    "748"
                ],
                "is_rel_cate4": [
                    0
                ],
                "country": [
                    ""
                ],
                "prop_seaon": [
                    "Summer"
                ],
                "prop_length": [
                    "Maxi"
                ],
                "prop_main_material": [
                    "Polyester"
                ],
                "prop_pattern": [
                    "Solid"
                ],
                "prop_style": [
                    "Sexy | Extravagant | Elegant"
                ],
                "prop_quantity": [
                    "1"
                ],
                "prop_fitness": [
                    "Regular Fit"
                ],
                "show_7d": [
                    56152
                ],
                "click_7d": [
                    1960
                ],
                "cart_7d": [
                    51
                ],
                "ord_total": [
                    363
                ],
                "pay_total": [
                    320
                ],
                "ord_7d": [
                    5
                ],
                "pay_7d": [
                    5
                ],
                "sales_price": [
                    0
                ],
                "ctr_7d": [
                    0.03490525715913948
                ],
                "cvr_7d": [
                    0.002551020408163265
                ]
            }
        ]
    }
    rec_col = {"signature_name": "serving_default", "inputs": {
        "highLevelSeqListGoods": [[
            "1327692",
            "1402902",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ]],
        "highLevelSeqListCateId": [[
            "748",
            "449",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ]],
        "lowerLevelSeqListGoods": [[
            "1327692",
            "1402902",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ]],
        "lowerLevelSeqListCateId": [[
            "748",
            "449",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ]],
        "register_brand": ["other"],
        "last_login_device":["huawei"],
        "last_login_brand":["huawei"],
        "main_goods_id": ["1402902"],
        "main_cate_id": [
            "449"
        ],
        "main_cate_level2_id": [
            "12"
        ],
        "main_cate_level3_id": [
            "74"
        ],
        "main_cate_level4_id": [
            "449"
        ],
        "goods_id": [
            ["1327692"]
        ],
        "cate_id": [
            "748"
        ],
        "is_rel_cate": [
            0
        ],
        "cate_level1_id": [
            "2"
        ],
        "cate_level2_id": [
            "12"
        ],
        "is_rel_cate2": [
            1
        ],
        "cate_level3_id": [
            "79"
        ],
        "is_rel_cate3": [
            0
        ],
        "cate_level4_id": [
            "748"
        ],
        "is_rel_cate4": [
            0
        ],
        "country": [
            ""
        ],
        "prop_seaon": [
            "Summer"
        ],
        "prop_length": [
            "Maxi"
        ],
        "prop_main_material": [
            "Polyester"
        ],
        "prop_pattern": [
            "Solid"
        ],
        "prop_style": [
            "Sexy | Extravagant | Elegant"
        ],
        "prop_quantity": [
            "1"
        ],
        "prop_fitness": [
            "Regular Fit"
        ],
        "show_7d": [
            56152
        ],
        "click_7d": [
            1960
        ],
        "cart_7d": [
            51
        ],
        "ord_total": [
            363
        ],
        "pay_total": [
            320
        ],
        "ord_7d": [
            5
        ],
        "pay_7d": [
            5
        ],
        "sales_price": [
            0
        ],
        "ctr_7d": [
            0.03490525715913948
        ],
        "cvr_7d": [
            0.002551020408163265
        ]
    }}

    if args.debug=='1':
        request['debug'] = "1"
    else:
        request['debug'] = ""

    if args.format == 'row':
        request['ipt'] = req_row
    elif args.format == 'col':
        request['ipt'] = rec_col

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
    print('end run pipeline:', args.pipeline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='deploy',
        description='deploy',
        epilog='deploy')
    parser.add_argument('--pipeline', default='pkg,edp,req_sg')
    parser.add_argument('--endpoint', default='prod-edp-model')
    parser.add_argument('--region', default='in')
    parser.add_argument('--edp_version', default='v1')
    parser.add_argument('--model_dir', default='prod_model/')
    parser.add_argument('--model_name', default='prod_mtl_seq_on_esmm_v1')
    parser.add_argument('--model_version', default='/ds=20241202-20241209/model/')
    parser.add_argument('--tar_name', default='prod_mtl_seq_on_esmm_v1_v1.tar.gz')
    parser.add_argument('--debug', default='1')
    parser.add_argument('--format', default='col')
    parser.add_argument('--instance_type', default='ml.r5.xlarge')
    args = parser.parse_args()
    args.endpoint = 'edp-' + args.model_name.replace('_', '-') + '-' + args.edp_version
    print('endpoint:', args.endpoint)
    main(args)
