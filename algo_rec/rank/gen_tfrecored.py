def bytes_fea(v_list, n=1, encode=False):
    v_list = v_list if isinstance(v_list, list) else [v_list]
    if len(v_list) > n:
        v_list = v_list[:n]
    elif len(v_list) < n:
        v_list.extend([" "] * (n - len(v_list)))
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[bytes(v, encoding="utf8") for v in v_list]))

def ints_fea(v_list):
    v_list = v_list if isinstance(v_list, list) else [v_list]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v or 0) for v in v_list]))

def floats_fea(v_list):
    v_list = v_list if isinstance(v_list, list) else [v_list]
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v or 0) for v in v_list]))

def cross_fea(v1_list, v2_list, n=1):
    v1_list = v1_list if isinstance(v1_list, list) else [v1_list]
    v2_list = v2_list if isinstance(v2_list, list) else [v2_list]
    # cross feature usuannly contains query or title.
    v3_list = ['%s,,%s' % (v1, v2) for v1 in v1_list for v2 in v2_list]
    return bytes_fea(v3_list, n, True)

def build_example(sample):
    query = sample['query'].lower()
    query_seg = query.split()
    title = sample['title'].lower()
    title_seg = title.split()

    feature = dict()
    feature.update({k: floats_fea(sample[k]) for k in ["ctr_7d", "cvr_7d"]})
    feature.update({k: bytes_fea(sample[k]) for k in ["cate_id", "goods_id", "cate_level1_id", "cate_level2_id", "cate_level3_id", "cate_level4_id","country"]})
    feature.update({k: bytes_fea(sample[k], n=20) for k in ["seq_cate_id", "seq_goods_id"]})
    feature.update({k: ints_fea(sample[k]) for k in ["show_7d", "click_7d", "cart_7d", "ord_total", "pay_total", "ord_7d","pay_7d"]})
    feature.update({k: ints_fea(sample[k]) for k in ["is_clk", "is_pay","show_7d", "click_7d", "cart_7d", "ord_total", "pay_total", "ord_7d","pay_7d"]})
    return tf1.train.Example(features=tf1.train.Features(feature=feature))

