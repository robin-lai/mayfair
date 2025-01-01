import os
import argparse
import time

def main(args):
    from_dir = "s3://warehouse-algo/rec/cn_rec_detail_sample_v20_savana_in_tfr/ds=%s/" % args.ds
    to_dir = "s3://warehouse-algo/rec/cn_rec_detail_sample_v20_savana_in_tfr/ds=20241210-20241216/"
    os.system("aws s3 cp --recursive %s  %s" % (from_dir, to_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='deploy',
        description='deploy',
        epilog='deploy')
    parser.add_argument('--range', type=str, default='20241211,20241212,20241213,20241214,20241215,20241216')
    args = parser.parse_args()
    for ds in args.range.split(','):
        st = time.time()
        args.ds = ds
        main(args)
        ed = time.time()
        print('cp %s cost %s' % (str(args.ds), str(ed-st)))