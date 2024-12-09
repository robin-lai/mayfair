# coding: utf-8

import json
import logging
import pickle
import os

base_data_dir = '/opt/ml/model/'
item_fts_file = base_data_dir + 'item_features.pkl'
logging.info('[DEBUG] current dir: %s %s', os.getcwd(), os.listdir("/opt/ml/model/"))
item_features_string = {"goods_id":"", "cate_id": "", "cate_level1_id":"","cate_level2_id":"", "cate_level3_id":"", "cate_level4_id":"", "country":""}
item_features_double = {"ctr_7d":0.0, "cvr_7d":0.0}
item_features_int = {"show_7d":0, "click_7d":0, "cart_7d":0, "ord_total":0,"pay_total":0,"ord_7d":0,"pay_7d":0 }
user_seq_string = {"seq_goods_id":[""] * 20, "seq_cate_id":[""] * 20}
user_seq_on_string = {"highLevelSeqListGoods":[""] * 20, "highLevelSeqListCateId":[""] * 20,
                      "lowerLevelSeqListGoods":[""] * 20, "lowerLevelSeqListCateId":[""] * 20}

def get_infer_json():
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
        "seq_goods_id": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                         "19", "20"],
        "goods_id": ["1"],
        "seq_cate_id": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                        "19", "20"],
        "cate_id": ["1"],
    }
    ipt = {"signature_name": "prediction","instances": [inputs_seq,inputs_seq] }
    return ipt

def request_check(d):
    if 'goodsIdList' not in d:
        msg = '[E] goodsIdList need'
        logging.ERROR(msg)
        raise ValueError(msg)
    if 'featureMap' not in d:
        msg = '[E] featureMap need'
        logging.ERROR(msg)
        raise ValueError(msg)
    return True



def get_infer_json_from_request(d):
    ipt = {"signature_name": "prediction"}
    ll = []
    if request_check(d):
        with open(item_fts_file, 'rb') as fin:
            m = pickle.load(fin)
        for goods_id in d['goodsIdList']:
            example = {}
            example['sample_id'] = 100
            for name in item_features_string.keys():
                if goods_id in m:
                    if name in m[goods_id]:
                        example[name] = [str(m[goods_id][name])]
                    else:
                        example[name] = [str(item_features_string[name])]
                else:
                    example[name] = [str(item_features_string[name])]
            for name in item_features_int.keys():
                if goods_id in m:
                    if name in m[goods_id]:
                        example[name] = [int(m[goods_id][name])]
                    else:
                        example[name] = [int(item_features_int[name])]
                else:
                    example[name] = [int(item_features_int[name])]
            for name in item_features_double.keys():
                if goods_id in m:
                    if name in m[goods_id]:
                        example[name] = [float(m[goods_id][name])]
                    else:
                        example[name] = [float(item_features_double[name])]
                else:
                    example[name] = [float(item_features_double[name])]
            for name in user_seq_on_string:
                if name == 'highLevelSeqListGoods':
                    seq = d['featureMap']['userFeatures']['high_level_seq']
                    if len(seq) == 20:
                        example['highLevelSeqListGoods'] = seq
                    else:
                        example['highLevelSeqListGoods'] = seq + [""] * (20-len(seq))

                if name == 'lowerLevelSeqListGoods':
                    seq = d['featureMap']['userFeatures']['high_level_seq']
                    if len(seq) == 20:
                        example['lowerLevelSeqListGoods'] = seq
                    else:
                        example['lowerLevelSeqListGoods'] = seq + [""] * (20-len(seq))

                if name == 'highLevelSeqListCateId':
                    cate_list = []
                    for e in d['featureMap']['userFeatures']['high_level_seq']:
                        if e in m:
                            cate_list.append(m[e]['cate_id'])
                        else:
                            cate_list.append("")
                            print('goods_id:%s not in item map'%e)
                    if len(cate_list) == 20:
                        example['highLevelSeqListCateId'] = cate_list
                    else:
                        example['highLevelSeqListCateId'] = cate_list + [""] * (20 - len(cate_list))

                if name == 'lowerLevelSeqListCateId':
                    cate_list = []
                    for e in d['featureMap']['userFeatures']['low_level_seq']:
                        if e in m:
                            cate_list.append(m[e]['cate_id'])
                        else:
                            cate_list.append("")
                            print('goods_id:%s not in item map'%e)
                    if len(cate_list) == 20:
                        example['lowerLevelSeqListCateId'] = cate_list
                    else:
                        example['lowerLevelSeqListCateId'] = cate_list + [""] * (20 - len(cate_list))
            ll.append(example)
        ipt["instances"] = ll
    return ipt

def input_handler(data, context):
    """Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    if context.request_content_type == "application/json":
        logging.info('[DEBUG] current dir: %s %s', os.getcwd(), os.listdir("/opt/ml/model/"))
        d = json.loads(data.read())
        logging.info('[DEBUG] request_data1: %s', d)
        if d["debug"] == '1':
            print('debug=1 json_data', d)
            logging.info('debug=1 json_data',d)
            return data
        elif d["debug"] == '2':
            json_data = get_infer_json()
            print('debug=2 json_data', json_data)
            logging.info('debug=2 json_data',json_data)
            return json.dumps(json_data).encode('utf-8')
        ipt = get_infer_json_from_request(d)
        logging.info('ipt data:%s', ipt)
        ipt_encode = json.dumps(ipt).encode('utf-8')
        return ipt_encode


def output_handler(data, context):
    logging.info('[DEBUG] output_data: %s %s  %s', type(data), data, context)
    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type