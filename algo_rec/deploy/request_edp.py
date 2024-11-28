import json
from constant import *
import boto3

inputs1 = {
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
    "seq_goods_id": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                     "19", "20"],
    "goods_id": ["1"],
    "seq_cate_id": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                    "19", "20"],
    "cate_id": ["1"]
}

inputs3 = {
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
ipt4 = {"signature_name": "predict", "instances": [inputs3, inputs3]}
sg_client = boto3.client("sagemaker-runtime")

if __name__ == '__main__':
    print('inp-json-dump', json.dumps(ipt4))

    res = sg_client.invoke_endpoint(
        EndpointName=endpoint,
        Body=json.dumps(ipt4),
        # Body=json.dumps(ipt4).encode('utf-8'),
        # .encode('utf-8'),
        # Body=ipt4,
        ContentType="application/json"
    )
    print(res["Body"].read())
    # res_json = json.loads(res["Body"].read())
    # print(res_json)
    # res_json['predictions'][0]['probabilities'][0]