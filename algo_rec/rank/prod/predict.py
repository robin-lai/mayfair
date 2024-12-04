import tensorflow as tf
import argparse
from pyarrow import parquet

tf.compat.v1.enable_eager_execution()

import math
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    c = math.ceil(len(lst) / n)
    for i in range(0, len(lst), c):
        yield lst[i:i + c]

def get_infer_tensor_dict(type=2):
    tensor_dict = {
        "cate_level1_id": tf.constant(["1"], dtype=tf.string),
        "cate_level2_id": tf.constant(["1"], dtype=tf.string),
        "cate_level3_id": tf.constant(["1"], dtype=tf.string),
        "cate_level4_id": tf.constant(["1"], dtype=tf.string),
        "country": tf.constant(["1"], dtype=tf.string),
        "ctr_7d": tf.constant([0.1], dtype=tf.float32),
        "cvr_7d": tf.constant([0.1], dtype=tf.float32),
        "show_7d": tf.constant([100], dtype=tf.int64),
        "click_7d": tf.constant([100], dtype=tf.int64),
        "cart_7d": tf.constant([100], dtype=tf.int64),
        "ord_total": tf.constant([100], dtype=tf.int64),
        "pay_total": tf.constant([100], dtype=tf.int64),
        "ord_7d": tf.constant([100], dtype=tf.int64),
        "pay_7d": tf.constant([100], dtype=tf.int64),
        "seq_goods_id": tf.constant(
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
             "19", "20"], dtype=tf.string),
        "goods_id": tf.constant(["1"], dtype=tf.string),
        "seq_cate_id": tf.constant(
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
             "19", "20"], dtype=tf.string),
        "cate_id": tf.constant(["1"], dtype=tf.string),

    },
    tensor_dict2 = {
        "cate_level1_id": tf.constant([["1"]], dtype=tf.string),
        "cate_level2_id": tf.constant([["1"]], dtype=tf.string),
        "cate_level3_id": tf.constant([["1"]], dtype=tf.string),
        "cate_level4_id": tf.constant([["1"]], dtype=tf.string),
        "country": tf.constant([["1"]], dtype=tf.string),
        "ctr_7d": tf.constant([[0.1]], dtype=tf.float32),
        "cvr_7d": tf.constant([[0.1]], dtype=tf.float32),
        "show_7d": tf.constant([[100]], dtype=tf.int64),
        "click_7d": tf.constant([[100]], dtype=tf.int64),
        "cart_7d": tf.constant([[100]], dtype=tf.int64),
        "ord_total": tf.constant([[100]], dtype=tf.int64),
        "pay_total": tf.constant([[100]], dtype=tf.int64),
        "ord_7d": tf.constant([[100]], dtype=tf.int64),
        "pay_7d": tf.constant([[100]], dtype=tf.int64),
        "seq_goods_id": tf.constant(
            [["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
             "19", "20"]], dtype=tf.string),
        "goods_id": tf.constant([["1"]], dtype=tf.string),
        "seq_cate_id": tf.constant(
            [["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
             "19", "20"]], dtype=tf.string),
        "cate_id": tf.constant([["1"]], dtype=tf.string),
        "sample_id": tf.constant([[2]], dtype=tf.int32),
    }
    tensor_dict3 = {
        "cate_level1_id": tf.constant([["1"],["1"]], dtype=tf.string),
        "cate_level2_id": tf.constant([["1"],["1"]], dtype=tf.string),
        "cate_level3_id": tf.constant([["1"],["1"]], dtype=tf.string),
        "cate_level4_id": tf.constant([["1"],["1"]], dtype=tf.string),
        "country": tf.constant([["1"],["1"]], dtype=tf.string),
        "ctr_7d": tf.constant([[0.1],[0.1]], dtype=tf.float32),
        "cvr_7d": tf.constant([[0.1],[0.1]], dtype=tf.float32),
        "show_7d": tf.constant([[100],[100]], dtype=tf.int64),
        "click_7d": tf.constant([[100],[100]], dtype=tf.int64),
        "cart_7d": tf.constant([[100],[100]], dtype=tf.int64),
        "ord_total": tf.constant([[100],[100]], dtype=tf.int64),
        "pay_total": tf.constant([[100],[100]], dtype=tf.int64),
        "ord_7d": tf.constant([[100],[100]], dtype=tf.int64),
        "pay_7d": tf.constant([[100],[100]], dtype=tf.int64),
        "seq_goods_id": tf.constant(
            [["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
              "19", "20"],["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
              "19", "20"]], dtype=tf.string),
        "goods_id": tf.constant([["1"],["1"]], dtype=tf.string),
        "seq_cate_id": tf.constant(
            [["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
              "19", "20"],["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
              "19", "20"]], dtype=tf.string),
        "cate_id": tf.constant([["1"],["1"]], dtype=tf.string),
        "sample_id": tf.constant([[2],[2]], dtype=tf.int32),
    }
    if type==1:
        return tensor_dict2
    elif type==2:
        return tensor_dict3

def process(*args):
    pt_file = args.pt_file
    pt = parquet.read_table(pt_file).to_pydict()
    n = pt['sample_id']
    ll = [i for i in range(n)]
    batch = chunks(ll, 1024)
    item_features_string = {"goods_id": "", "cate_id": "", "cate_level1_id": "", "cate_level2_id": "",
                            "cate_level3_id": "", "cate_level4_id": "", "country": ""}
    item_features_double = {"ctr_7d": 0.0, "cvr_7d": 0.0}
    item_features_int = {"show_7d": 0, "click_7d": 0, "cart_7d": 0, "ord_total": 0, "pay_total": 0, "ord_7d": 0,
                         "pay_7d": 0}
    user_seq_string = {"seq_goods_id": [""] * 20, "seq_cate_id": [""] * 20}
    ret = []
    predictor = tf.saved_model.load_v2(args.dir).signatures["serving_default"]
    def padding(l):
        if len(l) > 20:
            return l[0:20]
        else:
            return l + [""] * (20-len(l))
    for idx in batch:
        feed_dict = {}
        for name in item_features_string.keys():
            v =[ [str(i)] for i in  pt[name][idx[0]:idx[-1]]]
            feed_dict[name] = tf.constant(v, dtype=tf.string)
        for name in item_features_int.keys():
            v =[ [int(i)] for i in  pt[name][idx[0]:idx[-1]]]
            feed_dict[name] = tf.constant(v, dtype=tf.int32)
        for name in item_features_double.keys():
            v =[ [float(i)] for i in  pt[name][idx[0]:idx[-1]]]
            feed_dict[name] = tf.constant(v, dtype=tf.float32)
        for name in user_seq_string.keys():
            v =[ padding(i) for i in  pt[name][idx[0]:idx[-1]]]
            feed_dict[name] = tf.constant(v, dtype=tf.float32)
        print('feed_dict', feed_dict)
        res = predictor(**feed_dict)
        print('res', res)
        ret.append(res)

def main(*args):
    process(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='predict',
        description='predict',
        epilog='predict')
    parser.add_argument('--pt_file', default='s3://warehouse-algo/rec/cn_rec_detail_sample_v1/ds=20241112/part-00000-1186234f-fa44-44a8-9aff-08bcf2c5fb26-c000')
    parser.add_argument('--dir', default='/home/sagemaker-user/mayfair/algo_rec/model/model_local_pkg/1732159550')
    args = parser.parse_args()
    main(args)
