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
    parser.add_argument('--range', type=str, default='20250310,20250311,20250312,20250313,20250314,20250315,20250316,20250317,20250318,20250319,20250320,20250321,20250322,20250323')
    parser.add_argument('--from_dir', type=str, default='s3://warehouse-algo/rec/cn_rec_detail_sample_v30_savana_in_tfr_row_n300_sample_select/ds=%s/')
    parser.add_argument('--to_dir', type=str, default='s3://warehouse-algo/rec/cn_rec_detail_sample_v30_savana_in_tfr_row_n300_sample_select/ds=20250310-20250323/')
    args = parser.parse_args()
    main(args)
