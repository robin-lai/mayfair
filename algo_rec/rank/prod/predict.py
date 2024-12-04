import tensorflow as tf
tf.compat.v1.enable_eager_execution()

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
local_model_dir = '/home/sagemaker-user/mayfair/algo_rec/deploy/tmp/1733270759'
predictor = tf.saved_model.load_v2(local_model_dir).signatures["serving_default"]
tensor_dict = get_infer_tensor_dict(type=2)
print(predictor(**tensor_dict))
