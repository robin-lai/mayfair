import os
import argparse
import time
import multiprocessing

def process(from_dir, to_dir):
    os.system("aws s3 cp --recursive %s  %s" % (from_dir, to_dir))


def main(args):
    ll = [[args.from_dir % ds, args.to_dir] for ds in args.range.split(',')]
    proc_list = [multiprocessing.Process(target=process, args=t) for t in ll]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='file_copy',
        description='file_copy',
        epilog='file_copy')
    parser.add_argument('--range', type=str, default='20250129,20250130,20250131,20250201,20250202,20250203,20250204')
    parser.add_argument('--from_dir', type=str, default='s3://warehouse-algo/rec/cn_rec_detail_sample_v30_savana_in_tfr_row_n300/ds=%s/')
    parser.add_argument('--to_dir', type=str, default='s3://warehouse-algo/rec/cn_rec_detail_sample_v30_savana_in_tfr_row_n300/ds=20250129-20250204/')
    args = parser.parse_args()
    main(args)
