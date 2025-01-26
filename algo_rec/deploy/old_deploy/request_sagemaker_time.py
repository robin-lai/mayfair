import json
import time

import boto3
import argparse
import numpy as np

request = {"signature_name": "serving_default", "city": "Menbai", "country": "IN", "debug": "",
                "featureMap": {"userFeatures": {"high_level_seq": ["1327692"] * 20, "low_level_seq": ["1327692"] * 20}},
                "goodsIdList": ["1327692", "1402902"], "ip": "127.0.0.1", "platform": "H5", "province": "Menbai",
                "scene": "detail_rec", "userId": "23221", "userNo": "2321", "uuid": "fxleyu", "version": "8.2.2"}

sg_client = boto3.client("sagemaker-runtime")

def main(args):
    request['goodsIdList'] = ["1327692"] * args.goods_num
    result = []
    cost = []
    for i in range(args.n):
        print('req idx:',i)
        if i % 5 == 0:
            print('sleep 1s')
            time.sleep(1)
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
        cost.append(ed-st)
        res_json = json.loads(res["Body"].read())
        pred = res_json['predictions']
        for j in range(len(pred)):
            result.append(pred[j]["ctr"])
        if i == 0:
            print('result',result)
    print('req:', args.n)
    print('goods_id_num:', args.goods_num)
    print('mean cost:', np.mean(cost), 'max cost:', np.max(cost), 'min cost:', np.min(cost))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='sg',
        description='sg',
        epilog='sg')
    parser.add_argument('--goods_num', type=int, default=600)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--endpoint', type=str, default='edp-prod-mtl-seq-on-esmm-v0-v6')
    args = parser.parse_args()
    main(args)

