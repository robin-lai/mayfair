import os
import argparse
from datetime import datetime, date, timedelta


def main(args):
    for i in range(args.part):
        from_file = '%s/swing_rec_Savana_IN_part_%s' % (args.from_dir, i)
        os.system("aws s3 cp %s  %s" % (from_file, args.to_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='file_copy',
        description='file_copy',
        epilog='file_copy')
    parser.add_argument('--ds', type=str,
                        default=(datetime.today() - timedelta(days=1)).strftime('%Y%m%d'))
    parser.add_argument('--from_dir', type=str,
                        default='s3://warehouse-algo/rec/recall/cn_rec_detail_recall_i2i_for_redis_row_n300/item_user_debias_%s_1.0_0.6_0.5/')
    parser.add_argument('--to_dir', type=str, default='s3://algo-sg/rec/cn_rec_detail_recall_i2i_for_redis/')
    parser.add_argument('--part', type=int, default=10)
    args = parser.parse_args()
    args.from_dir = args.from_dir % args.ds
    print(args.from_dir)
    main(args)
