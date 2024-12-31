# coding: utf-8

import json
import logging
import pickle
import os
import time
import argparse

# base_data_dir = '/home/sagemaker-user/mayfair/algo_rec/deploy/prod_v1/pkg/'
base_data_dir = '/opt/ml/model/'
item_fts_file = base_data_dir + 'item_features.pkl'
# logging.info('[DEBUG] current dir: %s %s', os.getcwd(), os.listdir("/opt/ml/model/"))
item_features_string = {"goods_id": "", "cate_id": "", "cate_level1_id": "", "cate_level2_id": "",
                        "cate_level3_id": "",
                        "cate_level4_id": "", "country": "",
                        "prop_seaon": "", "prop_length": "", "prop_main_material": "", "prop_pattern": "",
                        "prop_style": "", "prop_quantity": "", "prop_fitness": ""}
item_features_double = {"ctr_7d": 0.0, "cvr_7d": 0.0}
item_features_int = {"show_7d": 0, "click_7d": 0, "cart_7d": 0, "ord_total": 0, "pay_total": 0, "ord_7d": 0,
                     "pay_7d": 0, "sales_price": 0}
user_seq_string = {"seq_goods_id": [""] * 20, "seq_cate_id": [""] * 20}
user_seq_on_string = {"highLevelSeqListGoods": [""] * 20, "highLevelSeqListCateId": [""] * 20,
                      "lowerLevelSeqListGoods": [""] * 20, "lowerLevelSeqListCateId": [""] * 20}
user_profile_string = {"last_login_device": "", "last_login_brand": "", "register_brand": "", "client_type":""}
context_related = {"is_rel_cate": 0, "is_rel_cate2": 0, "is_rel_cate3": 0, "is_rel_cate4": 0}
main_item = {"main_cate_id": "", "main_cate_level2_id": "", "main_cate_level3_id": "", "main_cate_level4_id": ""}

with open(item_fts_file, 'rb') as fin:
    item_dict = pickle.load(fin)
    print('item_dict num:', len(item_dict.keys()))

def request_check(d):
    if 'goodsIdList' not in d:
        msg = '[E] goodsIdList need'
        logging.ERROR(msg)
        raise ValueError(msg)
    if 'featureMap' not in d:
        msg = '[E] featureMap need'
        logging.ERROR(msg)
        raise ValueError(msg)
    if 'parentGoodsId' not in d:
        msg = '[E] parentGoodsId need'
        logging.ERROR(msg)
        raise ValueError(msg)
    return True


def get_infer_json_from_request(d):
    ipt = {"signature_name": "serving_default"}
    ll = []
    if request_check(d):
        st = time.time()

        ed = time.time()
        print('load item fts cost:', ed - st)
        st = time.time()
        example_base = {}
        for name in user_seq_on_string.keys():
            if name == 'highLevelSeqListGoods':
                seq = d['featureMap']['userFeatures']['high_level_seq']
                if len(seq) == 20:
                    example_base['highLevelSeqListGoods'] = seq
                    example_base['highLevelSeqList_len'] = [20]
                else:
                    example_base['highLevelSeqListGoods'] = seq + [""] * (20 - len(seq))
                    example_base['highLevelSeqList_len'] = [len(seq)]

            if name == 'lowerLevelSeqListGoods':
                seq = d['featureMap']['userFeatures']['low_level_seq']
                if len(seq) == 20:
                    example_base['lowerLevelSeqListGoods'] = seq
                    example_base['lowerLevelSeqList_len'] = [20]
                else:
                    example_base['lowerLevelSeqListGoods'] = seq + [""] * (20 - len(seq))
                    example_base['lowerLevelSeqList_len'] = [len(seq)]

            if name == 'highLevelSeqListCateId':
                cate_list = []
                for e in d['featureMap']['userFeatures']['high_level_seq']:
                    if e in item_dict:
                        cate_list.append(item_dict[e]['cate_id'])
                    else:
                        cate_list.append("")
                        print('goods_id:%s not in item map' % e)
                if len(cate_list) == 20:
                    example_base['highLevelSeqListCateId'] = cate_list
                else:
                    example_base['highLevelSeqListCateId'] = cate_list + [""] * (20 - len(cate_list))

            if name == 'lowerLevelSeqListCateId':
                cate_list = []
                for e in d['featureMap']['userFeatures']['low_level_seq']:
                    if e in item_dict:
                        cate_list.append(item_dict[e]['cate_id'])
                    else:
                        cate_list.append("")
                        print('goods_id:%s not in item map' % e)
                if len(cate_list) == 20:
                    example_base['lowerLevelSeqListCateId'] = cate_list
                else:
                    example_base['lowerLevelSeqListCateId'] = cate_list + [""] * (20 - len(cate_list))
        for name in user_profile_string.keys():
            user_d = d['featureMap']['userFeatures']['user_feature_context']
            if name in user_d:
                example_base[name] = [user_d[name]]
            else:
                example_base[name] = [user_profile_string[name]]

        # main item
        main_goods_id = str(d['parentGoodsId'])
        example_base["main_goods_id"] = [main_goods_id]
        if main_goods_id not in item_dict:
            print('main_goods_id:%s not in item_dict' % (str(main_goods_id)))
        for name in main_item.keys():
            name_suf = name.lstrip('main_')
            if main_goods_id in item_dict:
                if name_suf in item_dict[main_goods_id]:
                    example_base[name] = [str(item_dict[main_goods_id][name_suf])]
                else:
                    example_base[name] = [str(main_item[name])]
            else:
                example_base[name] = [str(main_item[name])]

        for goods_id in d['goodsIdList']:
            example = {}
            example.update(example_base)
            for name in item_features_string.keys():
                if goods_id in item_dict:
                    if name in item_dict[goods_id]:
                        fts_v = str(item_dict[goods_id][name])
                        example[name] = [fts_v]
                        if name == 'cate_id':
                            if fts_v == example_base['main_cate_id'][0]:
                                example['is_rel_cate'] = [1]
                            else:
                                example['is_rel_cate'] = [0]
                        elif name == 'cate_level2_id':
                            if fts_v == example_base['main_cate_level2_id'][0]:
                                example['is_rel_cate2'] = [1]
                            else:
                                example['is_rel_cate2'] = [0]
                        elif name == 'cate_level3_id':
                            if fts_v == example_base['main_cate_level3_id'][0]:
                                example['is_rel_cate3'] = [1]
                            else:
                                example['is_rel_cate3'] = [0]
                        elif name == 'cate_level4_id':
                            if fts_v == example_base['main_cate_level4_id'][0]:
                                example['is_rel_cate4'] = [1]
                            else:
                                example['is_rel_cate4'] = [0]
                    else:
                        example[name] = [str(item_features_string[name])]
                else:
                    example[name] = [str(item_features_string[name])]
                    example['is_rel_cate'] = [0]
                    example['is_rel_cate2'] = [0]
                    example['is_rel_cate3'] = [0]
                    example['is_rel_cate4'] = [0]
            for name in item_features_int.keys():
                if goods_id in item_dict:
                    if name in item_dict[goods_id]:
                        example[name] = [int(item_dict[goods_id][name])]
                    else:
                        example[name] = [int(item_features_int[name])]
                else:
                    example[name] = [int(item_features_int[name])]
            for name in item_features_double.keys():
                if goods_id in item_dict:
                    if name in item_dict[goods_id]:
                        example[name] = [float(item_dict[goods_id][name])]
                    else:
                        example[name] = [float(item_features_double[name])]
                else:
                    example[name] = [float(item_features_double[name])]

            ll.append(example)
        ipt["instances"] = ll
        ed = time.time()
        print('gen tensor dict:', ed - st)
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
        # logging.info('[DEBUG] current dir: %s %s', os.getcwd(), os.listdir("/opt/ml/model/"))
        d = json.loads(data.read())
        # logging.info('[DEBUG] request_data1: %s', d)
        if "debug" not in d:
            d["debug"] = ""
        if d["debug"] == '1':
            print('debug=1 json_data', d)
            # logging.info('debug=1 json_data',d["ipt"])
            return json.dumps(d['ipt']).encode('utf-8')
        st = time.time()
        ipt = get_infer_json_from_request(d)
        ed = time.time()
        print('feature_process cost:', ed - st)
        if d["debug"] == 'log':
            print('req_input', ipt)

        # logging.info('ipt data:%s', ipt)
        ipt_encode = json.dumps(ipt).encode('utf-8')
        return ipt_encode

def output_handler(data, context):
    # logging.info('[DEBUG] output_data: %s %s  %s', type(data), data, context)
    response_content_type = context.accept_header
    prediction = data.content
    # json.loads(prediction)
    # print('prediction', prediction)
    return prediction, response_content_type

def main(args):
    d = {
        "city": "Menbai",
        "country": "IN",
        "debug": "",
        "featureMap": {
            "userFeatures": {
                "high_level_seq": [
                    "1327692",
                    "1402902"
                ],
                "low_level_seq": [
                    "1327692",
                    "1402902"
                ],
                "user_feature_context": {"register_brand": "other", "age": "40","last_login_device":"huawei", "last_login_brand":"huawei"}
            },
            "itemContextFeature": {

                "goodis1": {
                    "matchType": "u2i,i2i"
                },
                "goods_id2": {
                    "matchType": "u2i,i2i"
                }
            }
        },
        "goodsIdList": [
            "1402902", "1327692"
        ],
        "ip": "127.0.0.1",
        "parentGoodsId": "1490152",
        "platform": "H5",
        "province": "Menbai",
        "scene": "detail_rec",
        "userId": "23221",
        "userNo": "2321",
        "uuid": "fxleyu",
        "version": "8.2.2"
    }
    # print(get_infer_json_from_request(d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='inference',
        description='inference',
        epilog='inference')
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    debug = args.debug
    main(args)
