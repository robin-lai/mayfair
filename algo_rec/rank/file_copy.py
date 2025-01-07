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
    parser.add_argument('--range', type=str, default='20241227,20241228,20241229,20241230,20241231,20250101,20250102,20250103,20250104,20250105')
    parser.add_argument('--from_dir', type=str, default='s3://warehouse-algo/rec/cn_rec_detail_sample_v20_savana_in_tfr/ds=%s/')
    parser.add_argument('--to_dir', type=str, default='s3://warehouse-algo/rec/cn_rec_detail_sample_v20_savana_in_tfr/ds=20241227-20250105/')
    args = parser.parse_args()
    main(args)
