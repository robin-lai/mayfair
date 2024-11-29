import json
from constant import *
import boto3

request = {
    "uuid": "xxxxxx",
    "userId": "xxxx",
    "userNo": "1111",
    "scene": "detail_rec",
    "country": "IN",
    "ip": "1111",
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
            "high_level_seq": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                               "17", "18", "19", "20"],
            "low_level_seq": []
        },
        "contextFeatures": {},
        "itemFeatures": {}
    }
}

request2 = {"city": "Menbai", "country": "IN", "debug": "",
            "featureMap": { "userFeatures": {"high_level_seq": [], "low_level_seq": []}},
            "goodsIdList": ["1327692", "1402902"], "ip": "127.0.0.1", "platform": "H5", "province": "Menbai",
            "scene": "detail_rec", "userId": "23221", "userNo": "2321", "uuid": "fxleyu", "version": "8.2.2"}

sg_client = boto3.client("sagemaker-runtime")

if __name__ == '__main__':
    print('inp-json-dump', json.dumps(request))
    # endpoint = 'ctr-model-debug1121'

    res = sg_client.invoke_endpoint(
        EndpointName=endpoint,
        Body=json.dumps(request2),
        # Body=json.dumps(ipt4).encode('utf-8'),
        # .encode('utf-8'),
        # Body=ipt4,
        ContentType="application/json"
    )
    print(res["Body"].read())
    # res_json = json.loads(res["Body"].read())
    # print(res_json)
    # res_json['predictions'][0]['probabilities'][0]
