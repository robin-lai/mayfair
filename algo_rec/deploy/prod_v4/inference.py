# coding: utf-8
import copy
import json
import logging
import pickle
import os, sys, traceback
import time
import argparse

# base_data_dir = '/home/sagemaker-user/mayfair/algo_rec/deploy/prod_v1/pkg/'
base_data_dir = '/opt/ml/model/'
item_fts_file = base_data_dir + 'item_features.pkl'
item_stat_fts_file = base_data_dir + 'item_stat_features.pkl'
# logging.info('[DEBUG] current dir: %s %s', os.getcwd(), os.listdir("/opt/ml/model/"))

itemContextMap = {"mt_i2i_main": 0, "mt_i2i_long": 0, "mt_i2i_short": 0, "mt_hot_i2leaf": 0, "mt_hot": 0, "mt_u2i_f": 0,
                  "mt": ""}
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
        # mt_context.update({k: [v] for k, v in itemContextMapAddAfter.items()})
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
                ]
            }
        },
        "goodsIdList": [
            "1402902", "1327692"
        ],
        "itemContextMap": {
            "1402902": {"s": "i2i_long,i2i_short,hot,hot_i2leaf,i2i_main","ss": "0.1,0.2,1,2,0.3"}, "1327692": {"s": "","ss":""}
        },
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
