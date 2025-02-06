import os

import tensorflow as tf
import argparse
import tensorflow.compat.v1 as v1
from random import shuffle
import boto3

print(tf.__version__)


def get_file_batch(args):
    s3_cli = boto3.client('s3')
    BUCKET = 'warehouse-algo'

    # get files
    paginator = s3_cli.get_paginator('list_objects_v2')
    page_iter = paginator.paginate(Bucket=BUCKET, Prefix=args.tfr_s3)
    file_list = [[v['Key'] for v in page.get('Contents', [])] for page in page_iter][0]
    file_list = ['s3://%s/%s' % (BUCKET, v) for v in file_list]
    # print('file list num in dir', len(file_list))
    shuffle(file_list)
    return file_list[0:args.sample_num]


def process(score, tfr_file):
    def gen_col(score, idx, key):
        id = idx[key].tolist()
        id = [round(e[0], 5) for e in id]
        if key in score:
            score[key].extend(id)
        else:
            score[key] = id

    def parse(data):
        feature_describe = {
            "mt_i2i_main_score": v1.FixedLenFeature(1, tf.float32, default_value=0.0),
            "mt_i2i_long_score": v1.FixedLenFeature(1, tf.float32, default_value=0.0),
            "mt_i2i_short_score": v1.FixedLenFeature(1, tf.float32, default_value=0.0)
        }
        features = tf.io.parse_single_example(data, features=feature_describe)
        return features

    local_file = './tmp/' + tfr_file.split('/')[-1]
    os.system('aws s3 cp %s %s' % (tfr_file, local_file))

    ds = tf.data.TFRecordDataset(local_file)
    ds = ds.map(parse).batch(args.batch_size)
    # print(list(ds.as_numpy_iterator())[0:args.n])
    for idx in ds.as_numpy_iterator():
        gen_col(score, idx, "mt_i2i_main_score")
        gen_col(score, idx, "mt_i2i_long_score")
        gen_col(score, idx, "mt_i2i_short_score")


def main(args):
    score = {}
    file_batch = get_file_batch(args)
    print(f"file_batch:{file_batch}")
    for tfrfile in file_batch:
        process(score, tfrfile)

    import numpy as np
    pp = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    for k, v in score.items():
        bins = [round(e, 5) for e in np.percentile(v, pp)]
        print(f"{k}:{bins}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gentfr',
        description='gentfr',
        epilog='gentfr-help')
    parser.add_argument('--tfr_s3', type=str,
                        default='rec/cn_rec_detail_sample_v30_savana_in_tfr_row_n300/ds=20250120/')
    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    main(args)
