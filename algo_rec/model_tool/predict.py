import tensorflow as tf
import argparse


def get_sample_batch(predictor, sample_batch, goods_id2props):
    input_dict = {k: [] for k in
                  'uuid country query query_seg goods_id cid1 cid2 cid3 cid4 retarget title title_seg high_level_seq low_level_seq attrs hist_cart hist_long rebuy price'.split()}
    input_dict.update({a + '_' + b: [] for a in ['query', 'query_seg', 'country']
                       for b in ['goods_id', 'cid1', 'cid2', 'cid3', 'cid4']})
    for sample in sample_batch:
        for k in ['uuid', 'country', ]:
            input_dict[k].append([sample[k]])
        for k in ['query', 'title']:
            input_dict[k].append([sample[k].lower()])
        for k in ['query_seg', 'title_seg']:
            v = sample.get(k, sample.get(k[:5]))
            input_dict[k].append(padding(v.lower().split(), 20))
            # input_dict[k].append(padding(sample[k].lower().split()))

        bhv = sample['user_bhv_seq']
        high = []
        low = []
        hist_long = []
        hist_cart = []
        goods_id = str(sample["goods_id"])
        if bhv.strip() != "":
            bhv = json.loads(bhv)
            if "highLevelSeqList" in bhv:
                high = bhv["highLevelSeqList"]
            if "lowerLevelSeqList" in bhv:
                low = bhv["lowerLevelSeqList"]
            high = [item.split("\u0001")[1] for item in high]
            low = [item.split("\u0001")[1] for item in low]
            cartSeqList = bhv["cartSeqList"] if "cartSeqList" in bhv else []
            clickSeqList = bhv["clickSeqList"] if "clickSeqList" in bhv else []
            hist_cart = [item.split("\u0001")[1] if "\u0001" in item else item for item in cartSeqList]
            hist_long = [item.split("\u0001")[1] if "\u0001" in item else item for item in clickSeqList]
        sample['high_level_seq'] = high
        sample['low_level_seq'] = low
        sample['hist_long'] = hist_long
        sample['hist_cart'] = hist_cart

        rebuy = 0
        if goods_id in hist_cart:
            rebuy = 1
        sample['rebuy'] = rebuy

        retarget = 0
        if goods_id in high or goods_id in low:
            retarget = 1
        sample['retarget'] = retarget
        attrs = []
        if goods_id in goods_id2props:
            attrs = goods_id2props[goods_id].split(" ")
        sample['attrs'] = attrs

        for k in ["high_level_seq", "low_level_seq", "attrs", ]:
            input_dict[k].append(padding(sample[k], 20))
        for k in ["hist_cart", "hist_long"]:
            input_dict[k].append(padding(sample[k], 40))

        for k in ['goods_id', 'cid1', 'cid2', 'cid3', 'cid4', 'retarget', 'rebuy', 'price']:
            v = sample[k]
            input_dict[k].append([int(v or 0)])
            if k in ['retarget', 'rebuy', 'price']:
                continue
            input_dict['query_' + k].append(['%s,,%s' % (sample['query'].lower(), v)])
            input_dict['country_' + k].append(['%s,,%s' % (sample['country'], v)])
            input_dict['query_seg_' + k].append(
                ['%s,,%s' % (term, v) for term in padding(sample['query'].lower().split())])
    tensor_dict = {name: tf.constant(value, dtype=tf.int64 if isinstance(value[0][0], int) else tf.string)
                   for name, value in input_dict.items()}
    return tensor_dict

def get_sample_test(batch_size):
    tensor_dict = {
        "ctr_7d": tf.constant([0.1], dtype=tf.float32,  name="ctr_7d"),
        "cvr_7d": tf.constant([0.1], dtype=tf.float32,  name="cvr_7d"),
        "show_7d": tf.constant([100], dtype=tf.int64,  name="show_7d"),
        "click_7d": tf.constant([100], dtype=tf.int64,  name="click_7d"),
        "cart_7d": tf.constant([100], dtype=tf.int64,  name="cart_7d"),
        "ord_total": tf.constant([100], dtype=tf.int64,  name="ord_total"),
        "pay_total": tf.constant([100], dtype=tf.int64,  name="pay_total"),
        "ord_7d": tf.constant([100], dtype=tf.int64,  name="ord_7d"),
        "pay_7d": tf.constant([100], dtype=tf.int64,  name="pay_7d"),
        "cate_id": tf.constant(["1"], dtype=tf.string,  name="cate_id"),
        "goods_id": tf.constant(["1"], dtype=tf.string,  name="goods_id"),
        "cate_level1_id": tf.constant(["1"], dtype=tf.string,  name="cate_level1_id"),
        "cate_level2_id": tf.constant(["1"], dtype=tf.string,  name="cate_level2_id"),
        "cate_level3_id": tf.constant(["1"], dtype=tf.string,  name="cate_level3_id"),
        "cate_level4_id": tf.constant(["1"], dtype=tf.string,  name="cate_level4_id"),
        "country": tf.constant(["IN"], dtype=tf.string,  name="country"),
        "seq_cate_id": tf.constant( ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
                                    ,dtype=tf.string, name="seq_cate_id"),
        "seq_goods_id": tf.constant( ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
                                     ,dtype=tf.string, name="seq_goods_id"),
        "is_clk": tf.constant([1], dtype=tf.int64, name="is_clk"),
        "is_pay": tf.constant([1], dtype=tf.int64, name="is_pay"),
    }
    tensor_list_dict = {
        "ctr_7d": tf.constant([[0.1],[0.1]], dtype=tf.float32, name="ctr_7d"),
        "cvr_7d": tf.constant([[0.1],[0.1]], dtype=tf.float32, name="cvr_7d"),
        "show_7d": tf.constant([[100],[100]], dtype=tf.int64, name="show_7d"),
        "click_7d": tf.constant([[100],[100]], dtype=tf.int64, name="click_7d"),
        "cart_7d": tf.constant([[100],[100]], dtype=tf.int64, name="cart_7d"),
        "ord_total": tf.constant([[100],[100]], dtype=tf.int64, name="ord_total"),
        "pay_total": tf.constant([[100], [100]], dtype=tf.int64, name="pay_total"),
        "ord_7d": tf.constant([[100],[100]], dtype=tf.int64, name="ord_7d"),
        "pay_7d": tf.constant([[100], [100]], dtype=tf.int64, name="pay_7d"),
        "cate_id": tf.constant([["1"], ["1"]], dtype=tf.string, name="cate_id"),
        "goods_id": tf.constant([["1"], ["1"]], dtype=tf.string, name="goods_id"),
        "cate_level1_id": tf.constant([["1"], ["1"]], dtype=tf.string, name="cate_level1_id"),
        "cate_level2_id": tf.constant([["1"], ["1"]], dtype=tf.string, name="cate_level2_id"),
        "cate_level3_id": tf.constant([["1"], ["1"]], dtype=tf.string, name="cate_level3_id"),
        "cate_level4_id": tf.constant([["1"], ["1"]], dtype=tf.string, name="cate_level4_id"),
        "country": tf.constant([["IN"],["IN"]], dtype=tf.string, name="country"),
        "seq_cate_id": tf.constant(
            [["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
            ,["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]], dtype=tf.string, name="seq_cate_id"),
        "seq_goods_id": tf.constant(
            [["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
            ,["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]], dtype=tf.string, name="seq_goods_id"),
        "is_clk": tf.constant([[1],[1]], dtype=tf.int64, name="is_clk"),
        "is_pay": tf.constant([[1], [1]], dtype=tf.int64, name="is_pay"),
    }
    if batch_size == 1:
        return tensor_dict
    elif batch_size > 1:
        return tensor_list_dict

def predict(args):
    tensor_dict = get_sample_test(args.batch_size)
    predictor = tf.saved_model.load_v2(args.local_model_dir + args.version).signatures[args.signatures]
    print("===========",tensor_dict)
    pred_batch = predictor(**tensor_dict)['probabilities']
    print(type(pred_batch))
    print('pred_batch:', pred_batch)
    print(tf.get_static_value(pred_batch))
    # print(pred_batch.tolist())
    dump_data = []
    # key_fields = ['click_label', 'pay_label', 'uuid', 'query', 'goods_id']
    # key_fields = sorted(list(sample.keys()))
    # for sample, pred in zip(sample_batch, pred_batch):
    #     line = [sample[field] for field in key_fields]
    #     line.insert(0, pred)
    #     dump_data.append('\t'.join(map(str, line)) + '\n')
    # return ''.join(dump_data)


def inference(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='predict',
        description='predict',
        epilog='predict-help')
    parser.add_argument('--local_model_dir', default='/home/sagemaker-user/mayfair/algo_rec/rank/exp/model_seq_nohead_1day/')
    parser.add_argument("--version", default="1732718918")
    parser.add_argument("--signatures", default="serving_default")
    parser.add_argument("--batch_size",type=int, default=1)
    args = parser.parse_args()
    predict(args)
