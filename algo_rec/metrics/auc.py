import pyarrow as pa
from pyarrow import parquet
from sklearn.metrics import roc_auc_score
import pickle
import argparse
import numpy as np
import sys
import traceback

# 方法1
def auc1(label, pre):
    pos = [i for i in range(len(label)) if label[i] == 1]
    neg = [i for i in range(len(label)) if label[i] == 0]
    auc = 0
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5
    if not pos or not neg:
        return None
    return auc / (len(pos) * len(neg))


# 方法2
def auc(label, pre):
    new_data = [[p, l] for p, l in zip(pre, label)]
    new_data.sort(key=lambda x: x[0])
    score_index = {}
    for index, i in enumerate(new_data):
        if i[0] not in score_index:
            score_index[i[0]] = []
        score_index[i[0]].append(index + 1)
    rank_sum = 0.
    for i in new_data:
        if i[1] == 1:
            rank_sum += sum(score_index[i[0]]) / len(score_index[i[0]]) * 1.0
    pos = label.count(1)
    neg = label.count(0)
    if not pos or not neg:
        return None
    return (rank_sum - (pos * (pos + 1) * 0.5)) / (pos * neg)

def gauc(pred,label_idx, pre_idx, type):
    gauc = {}
    gauc_l = []
    none_auc = 0
    try:
        for u, l in pred.items():
            pred = [e[pre_idx] for e in l]
            label = [e[label_idx] for e in l]
            auc_score = auc(label, pred)
            if auc_score is not None:
                gauc[u] = auc_score
                gauc_l.append(auc_score)
            else:
                none_auc += 1
                # print('uid:%s auc is none'%(u), l)
    except Exception:
        print('data:', l)
        traceback.print_exc(file=sys.stdout)
    print('none_auc num %s of all uuid:%s'%(str(none_auc),str(len(pred.keys()))))
    print('uuid num:%s have auc'%(str(len(gauc_l))))
    print('type:%s'%type, np.mean(gauc_l))
    pp = [10, 20, 30.40, 50, 60, 70, 80, 90, 100]
    print('type:%s percentle:'%type, np.percentile(gauc_l, pp))


def main(args):
    pred_file = "s3://warehouse-algo/rec/model_pred/prod_mtl_seq_on_esmm_v1"
    pt_file = './prod_mtl_seq_on_esmm_v1_pt_test.pkl'
    if args.debug:
        with open(pt_file, 'rb') as fin:
            pt = pickle.load(fin)
    else:
        pt = parquet.read_table(pred_file).to_pydict()
        n = 100000
        pt_test = {}
        for k in pt.keys():
            pt_test[k] = pt[k][0:n]
        with open(pt_file, 'wb') as fout:
            pickle.dump(pt_test, fout)

    pred = []
    uuid_pred = {}
    req_pred = {}
    for id, clk, pay, ctr, cvr in zip(pt['sample_id'], pt['is_clk'], pt['is_pay'], pt['ctr'], pt['cvr']):
        token = str(id).split('|')
        pred.append((token[0], token[2], clk, pay, ctr, cvr))
        tt = (clk, pay, ctr, cvr)
        if token[0] in uuid_pred:
            uuid_pred[token[0]].append(tt)
        else:
            uuid_pred[token[0]] = [tt]
        if token[2] in req_pred:
            req_pred[token[2]].append(tt)
        else:
            req_pred[token[2]] = [tt]

    print('uuid num:', len(uuid_pred.keys()))
    print('recid num:', len(req_pred.keys()))
    gauc(uuid_pred, 0,3, 'u-gauc')
    gauc(req_pred, 0,3, 'q-gauc')


# label = [0, 0, 1, 1, 1, 0]
# score = [0.1, 0.4, 0.35, 0.8, 0.8, 0.9]

# print(AUC1(label, score))
# print(AUC2(label, score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='auc',
        description='auc',
        epilog='auc')
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    main(args)


