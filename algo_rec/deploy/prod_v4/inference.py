# coding: utf-8
import copy
import json
import logging
import pickle
import os, sys, traceback
import time
import argparse

# base_data_dir = '/home/sagemaker-user/mayfair/algo_rec/deploy/mtl_seq_esmm_v4/pkg/'
base_data_dir = '/opt/ml/model/'
item_fts_file = base_data_dir + 'item_features.pkl'
item_stat_fts_file = base_data_dir + 'item_stat_features.pkl'
# logging.info('[DEBUG] current dir: %s %s', os.getcwd(), os.listdir("/opt/ml/model/"))

itemContextMap = {"mt_i2i_main": 0, "mt_i2i_long": 0, "mt_i2i_short": 0, "mt_hot_i2leaf": 0, "mt_hot": 0, "mt_u2i_f": 0}
mtFeatures = {"mt":[""] * 6}
recall_map = {"u2i_short": "mt_i2i_short", "u2i_long": "mt_i2i_long", "hot": "mt_hot", "hot_i2leaf": "mt_hot_i2leaf",
              "i2i_main": "mt_i2i_main", "u2i-f": "mt_u2i_f"}
itemContextRecallScore = {"mt_i2i_main_score": -1.0, "mt_i2i_long_score": -1.0, "mt_i2i_short_score": -1.0}

item_features_double = {'pctr_14d': -1, 'pcart_14d': -1,
                        'pwish_14d': -1, 'pcvr_14d': -1, 'pctr_30d': -1, 'pcart_30d': -1, 'pwish_30d': -1,
                        'pcvr_30d': -1}

item_features_string = {"goods_id": "", "cate_id": "", "cate_level2_id": "",
                        "cate_level3_id": "",
                        "cate_level4_id": ""}
item_features_int = {"sales_price": 0}

user_seq_on_string = {"highLevelSeqListGoods": [""] * 20, "highLevelSeqListCateId": [""] * 20,
                      "lowerLevelSeqListGoods": [""] * 20, "lowerLevelSeqListCateId": [""] * 20}

main_item = {"main_cate_id": "", "main_cate_level2_id": "", "main_cate_level3_id": "",
             "main_cate_level4_id": ""} # "main_goods_id" in code add
seq_len = 20

with open(item_fts_file, 'rb') as fin:
    item_dict = pickle.load(fin)
    print('item_dict num:', len(item_dict.keys()))

with open(item_stat_fts_file, 'rb') as fin:
    item_stat_dict = pickle.load(fin)
    print('item_stat_dict num:', len(item_stat_dict.keys()))


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
                seq_old = d['featureMap']['userFeatures']['high_level_seq']
                seq = [e for e in seq_old if e != '']
                if len(seq) >= seq_len:
                    example_base['highLevelSeqListGoods'] = seq[0:seq_len]
                    example_base['highLevelSeqList_len'] = [seq_len]
                else:
                    example_base['highLevelSeqListGoods'] = seq + [""] * (seq_len - len(seq))
                    example_base['highLevelSeqList_len'] = [len(seq)]

            if name == 'lowerLevelSeqListGoods':
                seq_old = d['featureMap']['userFeatures']['low_level_seq']
                seq = [e for e in seq_old if e != '']
                if len(seq) >= seq_len:
                    example_base['lowerLevelSeqListGoods'] = seq[0:seq_len]
                    example_base['lowerLevelSeqList_len'] = [seq_len]
                else:
                    example_base['lowerLevelSeqListGoods'] = seq + [""] * (seq_len - len(seq))
                    example_base['lowerLevelSeqList_len'] = [len(seq)]

            if name == 'highLevelSeqListCateId':
                cate_list = []
                seq_old = d['featureMap']['userFeatures']['high_level_seq']
                seq = [e for e in seq_old if e != '']
                for e in seq:
                    if e in item_dict:
                        cate_list.append(item_dict[e]['cate_id'])
                    else:
                        cate_list.append("")
                        print('goods_id:%s not in item map' % e)
                if len(cate_list) >= seq_len:
                    example_base['highLevelSeqListCateId'] = cate_list[0:seq_len]
                else:
                    example_base['highLevelSeqListCateId'] = cate_list + [""] * (seq_len - len(cate_list))

            if name == 'lowerLevelSeqListCateId':
                cate_list = []
                seq_old = d['featureMap']['userFeatures']['low_level_seq']
                seq = [e for e in seq_old if e != '']
                for e in seq:
                    if e in item_dict:
                        cate_list.append(item_dict[e]['cate_id'])
                    else:
                        cate_list.append("")
                        print('goods_id:%s not in item map' % e)
                if len(cate_list) >= seq_len:
                    example_base['lowerLevelSeqListCateId'] = cate_list[0:seq_len]
                else:
                    example_base['lowerLevelSeqListCateId'] = cate_list + [""] * (seq_len - len(cate_list))

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

        mt_context = {}
        mt_context.update({k: [v] for k, v in itemContextMap.items()})
        mt_context.update(mtFeatures)
        mt_context.update({k: [v] for k, v in itemContextRecallScore.items()})
        for goods_id in d['goodsIdList']:
            example = {}
            example.update(example_base)
            example.update(copy.deepcopy(mt_context))
            for name in item_features_string.keys():
                if goods_id in item_dict:
                    if name in item_dict[goods_id]:
                        fts_v = str(item_dict[goods_id][name])
                        example[name] = [fts_v]
                    else:
                        example[name] = [str(item_features_string[name])]
                else:
                    example[name] = [str(item_features_string[name])]
            for name in item_features_int.keys():
                if goods_id in item_stat_dict:
                    if name in item_stat_dict[goods_id]:
                        example[name] = [int(item_stat_dict[goods_id][name])]
                    else:
                        example[name] = [int(item_features_int[name])]
                else:
                    example[name] = [int(item_features_int[name])]
            for name in item_features_double.keys():
                if goods_id in item_stat_dict:
                    if name in item_stat_dict[goods_id]:
                        example[name] = [float(item_stat_dict[goods_id][name])]
                    else:
                        example[name] = [float(item_features_double[name])]
                else:
                    example[name] = [float(item_features_double[name])]

            if 'itemContextMap' in d and goods_id in d['itemContextMap']:
                s = d['itemContextMap'][goods_id]
                if s is not None and 's' in s:
                    s_str = s.get('s')
                    ss_str = s.get('ss')
                    if s_str != '' and ss_str != '':
                        s_ll = s_str.split(',')
                        if len(s_ll) < 6:
                            example['mt'] = list(s_ll).extend([""] * (6 - len(s_ll)))
                        else:
                            example['mt'] = list(s_ll)[0:6]
                        ss_ll = ss_str.split(',')
                        if len(s_ll) == len(ss_ll):
                            mt_d = {s_ll[i]: ss_ll[i] for i in range(len(s_ll))}
                            for k, v in mt_d.items():
                                example[recall_map[k]] = [1]
                                if recall_map[k].startswith('mt_i2i'):
                                    example[recall_map[k] + '_score'] = [float(v)]
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
        try:
            print('edp-version:0126')
            d = json.loads(data.read())
            print('request', d)
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
        except Exception:
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)


def output_handler(data, context):
    response_content_type = context.accept_header
    prediction = data.content
    try:
        print('response', json.loads(prediction))
    except Exception:
        print("-" * 60)
        logging.info('[DEBUG] output_data: %s %s  %s', type(data), data, context)
        logging.info('[DEBUG] output_data2: %s %s', response_content_type, prediction)
        traceback.print_exc(file=sys.stdout)
        print("-" * 60)
        # print('data:',t)
    return prediction, response_content_type


def main(args):
    d = {
        "signature_name": "serving_default",
        "city": "Menbai",
        "country": "IN",
        "debug": "",
        "parentGoodsId": "111307",
        "featureMap": {
            "userFeatures": {
                "high_level_seq": [
                    "1327692",
                    "1402902",
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
            "1402902",
            "1459992",
            "1477842","1540872","1301942","1321412","1333632","1319462"
        ],
        "itemContextMap": {
            "1327692": {"s": "u2i_long,u2i_short,hot,hot_i2leaf,i2i_main,u2i-f", "ss": "0.1,0.2,0.3,0.4,0.5,0.5"},
            "1402902": {"s": "", "ss": ""},
            "1459992": {"s": "u2i_long,u2i_short,hot,hot_i2leaf,i2i_main,u2i-f", "ss": "0.1,0.2,0.3,0.4,0.5,0.5"},
            "1477842": {"s": "", "ss": ""},
            '1538432': {'ss': '6122', 's': 'hot_i2leaf'}, '1483512': {'ss': '0.11273', 's': 'u2i_short'},
            '1319462': {'ss': '0.14308', 's': 'u2i_short'}, '1561212': {'ss': '0.00467', 's': 'i2i_main'},
            '1540872': {'ss': '0.01231,14207,88.0', 's': 'u2i_long,hot_i2leaf,hot'},
            '1611242': {'ss': '0.00761,22.0', 's': 'i2i_main,hot'}, '1570282': {'ss': '0.07823', 's': 'u2i_short'},
            '1436082': {'ss': '0.00767', 's': 'i2i_main'}, '1572232': {'ss': '5989', 's': 'hot_i2leaf'},
            '1487672': {'ss': '0.00783', 's': 'u2i_long'}, '1486462': {'ss': '0.00693', 's': 'i2i_main'},
            '1566462': {'ss': '0.08017,0.08017', 's': 'u2i_long,u2i_short'},
            '1234332': {'ss': '0.18611', 's': 'u2i_short'}, '1523292': {'ss': '5071', 's': 'hot_i2leaf'},
            '1495862': {'ss': '0.01243', 's': 'u2i_long'}, '1493442': {'ss': '9359', 's': 'hot_i2leaf'},
            '1301942': {'ss': '0.11918', 's': 'u2i_short'},
            '1263202': {'ss': '0.00956,8902', 's': 'u2i_long,hot_i2leaf'}, '1538782': {'ss': '5980', 's': 'hot_i2leaf'},
            '1321412': {'ss': '0.41866', 's': 'u2i_short'}, '1570612': {'ss': '0.00355,21.0', 's': 'i2i_main,hot'},
            '1472292': {'ss': '0.07618', 's': 'u2i_short'}, '1445252': {'ss': '5897', 's': 'hot_i2leaf'},
            '1333632': {'ss': '0.00712', 's': 'i2i_main'}, '116484': {'ss': '6196', 's': 'hot_i2leaf'}
        },
        "ip": "127.0.0.1",
        "platform": "H5",
        "province": "Menbai",
        "scene": "detail_rec",
        "userId": "23221",
        "userNo": "2321",
        "uuid": "fxleyu",
        "version": "8.2.2"
    }
    print(get_infer_json_from_request(d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='inference',
        description='inference',
        epilog='inference')
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    debug = args.debug
    main(args)
