import argparse
import os
import time

from pyarrow import parquet

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
        pt = parquet.read_table(args.u2i_s3).to_pylist()[0:1000000]
    else:
        pt = parquet.read_table(args.u2i_s3).to_pylist()
    for idx, d in enumerate(pt):
        if idx % 1000 == 0:
            print('process 1000')
        tl = []
        for id in d['goods_list']:
            i2i_k = 'Savana_IN' + chr(4) + str(id)
            if i2i_k in i2i_d:
                tgt_pair = i2i_d[i2i_k]
            else:
                continue
            for e in tgt_pair.split(chr(2)):
                tt = e.split(chr(4))
                if float(tt[1]) < 0.0000001:
                    continue
                tl.append((tt[0], float(tt[1])))
        if len(tl) < 1:
            continue
        tl.sort(key=lambda x: x[1], reverse=True)
        tl_set = set([str(e[0]) for e in tl])
        tl = list(tl_set)
        if args.debug:
            print(tl)
        kk = 'Savana_IN' + '|' + str(d['uuid'])
        if kk in u2i2i_d:
            u2i2i_d[kk].extend(tl[0:100] if len(tl) > 100 else tl)
        else:
            u2i2i_d[kk] = tl[0:100] if len(tl) > 100 else tl
    # line0, line1, line2 , line3 = [], [], [], []
    lines, fins = [[] for _ in range(args.part)], []
    for i in range(args.part):
        fin = open(args.u2i2i_file % i, 'w')
        fins.append(fin)

    # fin0 = open(args.u2i2i_file % 0, 'w')
    # fin1 = open(args.u2i2i_file % 1, 'w')
    # fin2 = open(args.u2i2i_file % 2, 'w')
    # fin3 = open(args.u2i2i_file % 3, 'w')
    for idx,k in enumerate(u2i2i_d.keys()):
        v = chr(2).join([str(e) + chr(4) + "1" for e in u2i2i_d[k]])
        line = (k + chr(1) + v + '\n')
        lines[idx%args.part].append(line)
        # if i%4 == 0: line0.append(line)
        # if i%4 == 1: line1.append(line)
        # if i%4 == 2: line2.append(line)
        # if i%4 == 3: line3.append(line)
    for i in range(args.part):
        fins[i].writelines(lines[i])
    # fin0.writelines(line0)
    # fin1.writelines(line1)
    # fin2.writelines(line2)
    # fin3.writelines(line3)
    for i in range(args.part):
        fins[i].close()
    # fin0.close()
    # fin1.close()
    # fin2.close()
    # fin3.close()
    for i in range(args.part):
        os.system("aws s3 cp %s %s" % (args.u2i2i_file % i, args.u2i2i_s3))
    # os.system("aws s3 cp %s %s" % (args.u2i2i_file % 1, args.u2i2i_s3))
    # os.system("aws s3 cp %s %s" % (args.u2i2i_file % 2, args.u2i2i_s3))
    # os.system("aws s3 cp %s %s" % (args.u2i2i_file % 3, args.u2i2i_s3))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='recall_u2i2i',
        description='recall_u2i2i',
        epilog='recall_u2i2i')
    parser.add_argument('--i2i_file', default='./swing_rec_Savana_IN_part_0_online,./swing_rec_Savana_IN_part_1_online')
    parser.add_argument('--u2i_s3', default='s3://warehouse-algo/rec/recall/cn_rec_detail_recall_wish_cart2i/ds=20241224/')
    parser.add_argument('--u2i2i_file', default='u2i2i_part_%s')
    parser.add_argument('--u2i2i_s3', default='s3://algo-sg/rec/recall_u2i2i/')
    parser.add_argument('--part',type=int, default=10)
    parser.add_argument('--debug', type=bool,  default=False)
    args = parser.parse_args()
    st = time.time()
    main(args)
    ed = time.time()
    print('cost:', ed-st)
