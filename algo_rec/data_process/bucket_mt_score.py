import os

import tensorflow as tf
import argparse
import tensorflow.compat.v1 as v1

print(tf.__version__)

def main(args):
    def gen_col(score, idx, key):
        id = idx[key].tolist()
        id = [e[0] for e in id]
        if key in score:
            score[key].extend(id)
        else:
            score[key] = id

    def parse(data):
        feature_describe = {
            "mt_i2i_main_score": v1.FixedLenFeature(1, tf.float32, default_value=0.0),
            "mt_i2i_long_score": v1.FixedLenFeature(1, tf.float32, default_value=0.0)
        }
        features = tf.io.parse_single_example(data, features=feature_describe)
        return features
    local_file = './tmp/'  + args.file.split('/')[-1]
    os.system('aws s3 cp %s %s' % (args.file, local_file))

    ds = tf.data.TFRecordDataset(local_file)
    ds = ds.map(parse).batch(args.batch_size)
    print(list(ds.as_numpy_iterator())[0:args.n])
    score = {}
    for idx in ds.as_numpy_iterator():
        gen_col(score, idx, "mt_i2i_main_score")
    import numpy as np
    pp = [5, 10, 15, 20,25,  30, 35, 40, 45, 50,55, 60,65, 70,75, 80,85, 90,95, 100]
    for k, v in score.items():
        print(f"{k}:{np.percentile(v, pp)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gentfr',
        description='gentfr',
        epilog='gentfr-help')
    parser.add_argument('--file',type=str, default='s3://warehouse-algo/rec/cn_rec_detail_sample_v30_savana_in_tfr_row_n300/ds=20250120/part-00000-7c77b5e6-3306-42b4-a878-1a0712ce3ab4-c000')
    parser.add_argument('--n',type=int, default=10)
    parser.add_argument('--batch_size',type=int, default=10)
    args = parser.parse_args()
    main(args)



