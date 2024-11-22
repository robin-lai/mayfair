# coding: utf-8

import json
import logging
import pickle
import os

logging.info('[DEBUG] current dir: %s %s', os.getcwd(), os.listdir("/opt/ml/model/"))

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
# inputs3 = {"a":"b"}

# ipt4 = {"signature_name": "serving_default","instances": [inputs3,inputs3] }
# ipt4 = {"signature_name": "serving_default","inputs": [inputs3,inputs3] }
ipt4 = {"signature_name": "predict","instances": [inputs3,inputs3] }

def input_handler(data, context):
    """Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    if context.request_content_type == "application/json":
        # logging.info('[DEBUG] request_data1: %s %s', context.request_content_type, '')
        # d = data.read().decode('utf-8')
        logging.info('[DEBUG] request_data2: %s %s', context.request_content_type, data)
        # d = data.read().decode('utf-8')
        new_data = json.dumps(ipt4).encode('utf-8')
        return new_data

    # raise ValueError('Invalid req type: %s' % context.request_content_type)


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    logging.info('[DEBUG] output_data: %s %s  %s', type(data), data, context)
    # logging.info('[DEBUG] output_data_content: %s', data.content.decode('utf-8'))
    # if data.status_code != 200:
    #     raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type