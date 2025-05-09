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
    # print(f"pos num:{pos} neg num:{neg}")
    if not pos or not neg:
        return None
    return (rank_sum - (pos * (pos + 1) * 0.5)) / (pos * neg)

def gauc(pred_d,label_idx, pre_idx, type):
    gauc = {}
    gauc_l = []
    none_auc = 0
    try:
        for u, l in pred_d.items():
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
    print('none_auc num %s of all %s :%s'%(str(none_auc),type, str(len(pred_d.keys()))))
    print('%s num:%s have auc'%(type, str(len(gauc_l))))
    print('type:%s'%type, np.mean(gauc_l))
    pp = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    print('type:%s percentle:'%type, np.percentile(gauc_l, pp))


def calc_auc(args):
    # label = [0, 0, 0, 0 ]
    # pred = [0.1, 0.2, 0.3, 0.4]
    # print(f"label all neg: auc {auc(label, pred)}")
    # label = [1, 1, 1, 1]
    # pred = [0.1, 0.2, 0.3, 0.4]
    # print(f"label all pos: auc {auc(label, pred)}")
    # label = [0, 1, 1, 1]
    # pred = [0.1, 0.2, 0.3, 0.4]
    # print(f"label have pos and neg: auc {auc(label, pred)}")
    label = [0, 1, 0, 1]
    pred = [0.1, 0.2, 0.3, 0.4]
    print(f"label have pos and neg: auc {auc(label, pred)}")
    # label = [0, 0, 0, 1]
    # pred = [0.9, 0.8, 0.7, 0.4]
    # print(f"label have pos and neg: auc {auc(label, pred)}")
    # label = [0, 0, 0, 1]
    # pred = [0.0, 0.1, 0.2, 0.9]
    # print(f"label have pos and neg: auc {auc(label, pred)}")


def main(args):
    pred_file = "s3://warehouse-algo/rec/model_pred/" + args.file
    pt_file = './%s_test.pkl' % (args.file)
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
        if args.sample == 'v10':
            token = str(id).split('|')
            uuid, reqid = token[0], token[2]
        elif args.sample == 'v20':
            # concat(
            #     bhv.country,
            #     '|',
            #     bhv.scene_code,
            #     '|',
            #     bhv.client_type,
            #     '|',
            #     bhv.uuid,
            #     '|',
            #     bhv.pssid,
            #     '|',
            #     bhv.recid,
            #     '|',
            #     bhv.main_goods_id,
            #     '|',
            #     bhv.goods_id
            # )
            # AS
            # sample_id,
            token = str(id).split('|')
            uuid, reqid = token[3], token[5]

        pred.append((uuid, reqid, clk, pay, ctr, cvr))
        tt = (clk, pay, ctr, cvr)
        if uuid in uuid_pred:
            uuid_pred[uuid].append(tt)
        else:
            uuid_pred[uuid] = [tt]
        if reqid in req_pred:
            req_pred[reqid].append(tt)
        else:
            req_pred[reqid] = [tt]

    label = [e[2] for e in pred]
    pre = [e[4] for e in pred]
    auc_all = auc(label, pre)
    print('N:', len(pred), 'label_mean:', np.mean(label), 'pred_mean:', np.mean(pre), 'auc-all-ctr:',auc_all)
    label_cvr = [e[3] for e in pred if e[2] == 1]
    pre_cvr = [e[5] for e in pred if e[2] == 1]
    auc_all_cvr = auc(label_cvr, pre_cvr)
    print('N:', len(label_cvr), 'label_mean:', np.mean(label_cvr), 'pred_mean:', np.mean(pre_cvr), 'auc-all-ctr:',auc_all_cvr)
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
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--sample', type=str, default='v10')
    args = parser.parse_args()
    # calc_auc(args)
    main(args)


