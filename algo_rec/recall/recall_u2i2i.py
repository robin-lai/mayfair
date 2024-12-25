import argparse
from pyarrow import parquet

from algo_rec.rank.prod.predict_tfr_mtl import debug


def main(args):
    # u2i_d = {}
    i2i_d = {}
    u2i2i_d = {}

    i2i_file_ll = args.i2i_file.split(',')
    for file in i2i_file_ll:
        with open(file, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                k, v = line.split(chr(1))
                i2i_d[k] = v
    print('read i2i end, num:', len(i2i_d.keys()))

    if args.debug:
        pt = parquet.read_table(args.u2i_s3).to_pylist()[0:100]
    else:
        pt = parquet.read_table(args.u2i_s3).to_pylist()
    for d in pt:
        tl = []
        for id in d['goods_list']:
            i2i_k = 'Savana_IN' + chr(1) + str(id)
            if i2i_k in i2i_d:
                tgt_pair = i2i_d[i2i_k]
            else:
                continue
            for e in tgt_pair.split(chr(2)):
                tt = e.split(chr(4))
                tl.append((tt[0], float(tt[1])))

        tl.sort(key=lambda x: x[1], reverse=True)
        if args.debug:
            print(tl)
        kk = 'Savana_IN' + '|' + str(d['uuid'])
        if kk in u2i2i_d:
            u2i2i_d[kk].extend(tl[0:100] if len(tl) > 100 else tl)
        else:
            u2i2i_d[kk] = tl[0:100] if len(tl) > 100 else tl
    line0, line1, line2 , line3 = [], [], [], []
    fin0 = open(args.u2i2i_file % 0, 'w')
    fin1 = open(args.u2i2i_file % 1, 'w')
    fin2 = open(args.u2i2i_file % 2, 'w')
    fin3 = open(args.u2i2i_file % 3, 'w')
    for i, kv in enumerate(u2i2i_d.items()):
        k = kv[0]
        v = chr(2).join([str(e[0]) + chr(4) + str(e[1]) for e in kv[1]])
        line = (k + chr(1) + v + '\n')
        if i%4 == 0: line0.append(line)
        if i%4 == 1: line1.append(line)
        if i%4 == 2: line2.append(line)
        if i%4 == 3: line3.append(line)
    fin0.writelines(line0)
    fin1.writelines(line1)
    fin2.writelines(line2)
    fin3.writelines(line3)
    fin0.close()
    fin1.close()
    fin2.close()
    fin3.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='recall_u2i2i',
        description='recall_u2i2i',
        epilog='recall_u2i2i')
    parser.add_argument('--i2i_file', default='./swing_rec_Savana_IN_part_0_online,./swing_rec_Savana_IN_part_1_online')
    parser.add_argument('--u2i_s3', default='s3://warehouse-algo/rec/recall/cn_rec_detail_recall_wish_cart2i/ds=20241224/')
    parser.add_argument('--u2i2i_file', default='u2i2i_part_%s')
    parser.add_argument('--part',type=int, default=4)
    parser.add_argument('--debug', type=bool,  default=False)
    args = parser.parse_args()
    main(args)
