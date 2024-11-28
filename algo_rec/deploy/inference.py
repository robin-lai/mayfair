# coding: utf-8

import json
import logging
import pickle
import os

logging.info('[DEBUG] current dir: %s %s', os.getcwd(), os.listdir("/opt/ml/model/"))

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
	"is_clk":[1],
	"is_pay":[1]
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
# inputs3 = {"a":"b"}

# ipt4 = {"signature_name": "serving_default","instances": [inputs3,inputs3] }
# ipt4 = {"signature_name": "serving_default","inputs": [inputs3,inputs3] }
ipt4 = {"signature_name": "prediction","instances": [inputs_no_seq,inputs_no_seq] }

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
        logging.info('[DEBUG] request_data1: %s %s', context.request_content_type, data)
        d = data.read().decode('utf-8')
        logging.info('[DEBUG] request_data2: %s', d)

        new_data = json.dumps(ipt4).encode('utf-8')
        return new_data

    # raise ValueError('Invalid req type: %s' % context.request_content_type)


def output_handler(data, context):
    logging.info('[DEBUG] output_data: %s %s  %s', type(data), data, context)
    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type