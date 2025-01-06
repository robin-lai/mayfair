import os
import argparse

import tensorflow as tf
import pandas as pd
import numpy as np


def main(args):
    s3_model = 's3://warehouse-algo/rec/prod_model/%s/%s/model/' % (args.model_name, args.model_version)
    local_model = './%s/%s/' % (args.model_name, args.model_version)
    os.system('aws s3 cp --recursive %s %s'%(s3_model, local_model))

    emb_name = ['attention_seq_on_high_cate_id/att_seq_on_high_cate_id/emb_att_seq_on_high_cate_id',
                'attention_seq_on_high_goods_id/att_seq_on_high_goods_id/emb_att_seq_on_high_goods_id'
        , 'attention_seq_on_low_cate_id/att_seq_on_low_cate_id/emb_att_seq_on_low_cate_id',
                'attention_seq_on_low_goods_id/att_seq_on_low_goods_id/emb_att_seq_on_low_goods_id']
    print('list variable:', tf.train.list_variables(local_model))

    def norm(arr):
        norm = np.linalg.norm(arr,ord=1, axis=1)
        norm_v = [e for e in norm if e != 0]
        norm_v.sort(reverse=True)
        print('norm1 num:', len(norm_v))
        print('norm1_s', norm_v[0:100])

        norm = np.linalg.norm(arr, ord=2, axis=1)
        norm_v = [e for e in norm if e != 0]
        norm_v.sort(reverse=True)
        print('norm2 num:', len(norm_v))
        print('norm2_s', norm_v[0:100])

    for var in emb_name:
        emb = tf.train.load_variable(local_model, var)
        print(emb.shape)
        # print(emb[0:100])
        print('var name:', var)
        norm(emb)
        emb_df = pd.DataFrame(emb)
        file_name = './' + var.split('/')[-1] + '.csv'
        emb_df.to_csv(file_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='check_model',
        description='check_model',
        epilog='check_model')
    parser.add_argument('--model_name', default='prod_mtl_seq_on_esmm_v20_mask_savana_in_fix')
    parser.add_argument('--model_version', default='ds=20241217-20241226')
    args = parser.parse_args()
    main(args)
