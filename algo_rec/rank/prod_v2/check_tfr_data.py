import os

import tensorflow as tf
import argparse
import tensorflow.compat.v1 as v1
from run_rec_model import _parse_fea

print(tf.__version__)

def main(args):

    local_file = args.file.split('/')[-1]
    os.system('aws s3 cp %s %s' % (args.file, local_file))
    ds = tf.data.TFRecordDataset(local_file)
    ds = ds.map(_parse_fea).batch(args.batch_size)
    print(list(ds.as_numpy_iterator())[0:args.n])
    # [{'is_clk': array([[1],
    #          [1]]),
    #   'is_pay': array([[0],
    #          [0]])},
    #  {'is_clk': array([[1],
    #          [1]]),
    #   'is_pay': array([[0],
    #          [0]])},

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gentfr',
        description='gentfr',
        epilog='gentfr-help')
    parser.add_argument('--file',type=str, default='s3://warehouse-algo/rec/cn_rec_detail_sample_v30_savana_in_tfr/ds=20250119/part-00000-54de2875-3f5d-4542-aeaf-07d25d9d5be7-c000')
    parser.add_argument('--n',type=int, default=10)
    parser.add_argument('--batch_size',type=int, default=10)
    args = parser.parse_args()
    main(args)



