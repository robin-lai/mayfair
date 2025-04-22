# encoding:utf-8
import argparse
from pyarrow import parquet
import numpy as np
import pickle

def main(args):
    pt = parquet.read_table(args.in_file)
    m = {}
    for uuid, goods_id, clk_num, country_code in zip(pt['uuid'], pt['goods_id'], pt['clk_num'], pt['country_code'][0:100000]):
        id = int(goods_id.as_py())
        u = uuid.as_py()
        if id in m:
            m[id].add(u)
        else:
            m[id] = set(u)
    ret = {}
    ll = m.keys()
    n = len(ll)
    for i in range(n-1):
        for j in range(i+1, n):
            s = m[i] & m[j]
            if s > 0:
                ret[str(ll[i]) + '_' + str(ll[j])] = [len(s), s]
    with open(args.out_pkl, 'wb') as fout:
        pickle.dump(ret, fout)

    print(np.percentile([v[0] for v in ret.values()], [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))

    d = {k: 10 for k in [10, 20, 30, 40, 50, 60, 70, 80, 90 , 100, 200, 300, 400, 500, 600]}
    c = len(d.keys()) * 10
    for k, v in ret.items():
        if v[0] in d and d[v[0]] > 0:
            print(f"itm1_itm2:{k} num:{v[0]}")
            d[v[0]] = d[v[0]] - 1
            c -= 1
            if c == 0:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='cooc',
        description='cooc',
        epilog='cooc')
    parser.add_argument('--in_file', type=str, default='s3://warehouse-algo/rec/cn_rec_detail_recall_ui_relation%s/ds=20240421')
    parser.add_argument('--out_pkl', type=str, default='swing_cooccurrence_num.pkl')
