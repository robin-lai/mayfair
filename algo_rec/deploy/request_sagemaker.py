import json
from constant import *
import boto3

request = {
    "uuid": "",
    "userId": "",
    "userNo": "",
    "scene": "detail_rec",
    "country": "",
    "ip": "",
    "debug": "",
    "city": "",
    "province": "",
    "platform": "",
    "version": "",
    "extMap": {
    },
    "goodsIdList": ["1", "2"],
    "featureMap": {
        "userFeatures": {
            "high_level_seq": ["1"] * 20,
            "low_level_seq": []
        },
        "contextFeatures": {},
        "itemFeatures": {}
    }
}

sg_client = boto3.client("sagemaker-runtime")

if __name__ == '__main__':
    print('inp-json-dump', json.dumps(request))
    # endpoint = 'ctr-model-debug1121'

    res = sg_client.invoke_endpoint(
        EndpointName=endpoint,
        Body=json.dumps(request),
        # Body=json.dumps(ipt4).encode('utf-8'),
        # .encode('utf-8'),
        # Body=ipt4,
        ContentType="application/json"
    )
    print(res["Body"].read())
    # res_json = json.loads(res["Body"].read())
    # print(res_json)
    # res_json['predictions'][0]['probabilities'][0]
