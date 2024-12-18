import argparse
import os
import pickle


def recall_i2i(args):
    i2i = {}
    if args.sync:
        for i in range(args.file_num):
            file = args.pattern % (str(i))
            s3_file = args.s3_dir + file
            local_file = './' + file
            os.system('aws s3 cp %s %s' % (s3_file, local_file))
            with open(local_file, 'r') as fin:
                lines = fin.readlines()
                for line in lines:
                    tk = line.split(chr(1))
                    tt = tk[0].split(chr(4))
                    tmp = {}
                    for fts in tk[1].split(chr(2)):
                        kk = fts.split(chr(4))
                        tmp.update({kk[0]:kk[1]})
                    i2i[tt[1]] = tmp
        with open(args.pkl_file, 'wb') as fout:
            pickle.dump(i2i, fout)
        print('sync file success and write as pkl file')
        print('swing i2i recall trig item num:', len(i2i.keys()))
    with open(args.pkl_file, 'rb') as fin:
        i2i = pickle.load(fin)
    if args.goods != '':
        for id in args.goods.split(','):
            if id in i2i:
                print('goods_id:%s ,'%(str(id)), i2i[id])
            else:
                print('goods_id:%s'%(str(id)), 'not in i2i')



def main(args):
    recall_i2i(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='recall_ana',
        description='recall_ana',
        epilog='recall_ana')
    parser.add_argument('--goods', type=str, default='')
    parser.add_argument('--sync', type=bool, default=False)
    parser.add_argument('--s3_dir', type=str, default='s3://algo-sg/rec/cn_rec_detail_recall_i2i_for_redis/')
    parser.add_argument('--pkl_file', type=str, default='./swing_rec_savana_in_i2i.pkl')
    parser.add_argument('--pattern', type=str, default='swing_rec_Savana_IN_part_%s')
    parser.add_argument('--file_num', type=int, default=10)
    args = parser.parse_args()
    print(args.sync)
    main(args)

