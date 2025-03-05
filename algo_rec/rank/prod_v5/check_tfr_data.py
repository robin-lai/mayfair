import os

import tensorflow as tf
import argparse
print(tf.__version__)
import tensorflow.compat.v1 as v1
from feature_serv_describe import feature_describe, feature_spec_serve

def main(args):
    def parse(data):
        features = tf.io.parse_single_example(data, features=feature_describe)
        return features
    s3_file = 's3://warehouse-algo/rec/cn_rec_detail_sample_v30_savana_in_tfr_row_n300_sample_select/ds=%s/%s' % (args.ds, args.file)
    local_file = './' + args.file
    os.system('aws s3 cp %s %s' % (s3_file, local_file))
    ds = tf.data.TFRecordDataset(local_file)
    ds2 = ds.map(parse).shuffle(args.batch_size * 10).batch(args.batch_size)
    ll = list(ds2.as_numpy_iterator())
    for idx, it in enumerate(ll):
        if args.names != '':
            for name in args.names.split(','):
                print('feature_name:', name)
                print(it[name])
        else:
            print(it)
        if idx == args.head:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='predict',
        description='predict',
        epilog='predict')
    parser.add_argument('--file', default='part-00000-5d8949ad-f96a-4927-9883-57e2e4023168-c000')
    parser.add_argument('--ds', default='20250304')
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--head',type=int, default=10)
    parser.add_argument('--names', default='')
    args = parser.parse_args()
    main(args)


