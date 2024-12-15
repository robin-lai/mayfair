import pyarrow as pa
from pyarrow import parquet
from sklearn.metrics import roc_auc_score
import pickle


pred_file = "s3://warehouse-algo/rec/model_pred/prod_mtl_seq_on_esmm_v1"
pt_all = parquet.read_table(pred_file).to_pydict()
n = 100000
pt = {}
for k in pt_all.keys():
    pt[k] = pt_all[k][0:n]
pt_file = './prod_mtl_seq_on_esmm_v1_pt_test.pkl'
with open(pt_file, 'wb') as fout:
    pickle.dump(pt, fout)

# with open(pt_file, 'rb') as fin:
#     pt = pickle.load(fin)


# 方法1
def AUC1(label, pre):
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
def AUC2(label, pre):
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


# label = [0, 0, 1, 1, 1, 0]
# score = [0.1, 0.4, 0.35, 0.8, 0.8, 0.9]

# print(AUC1(label, score))
# print(AUC2(label, score))
