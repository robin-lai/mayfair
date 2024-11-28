
import boto3
import sagemaker
from sagemaker import get_execution_role
# import sh
import os
from constant import *


def convert_text2pkl(text_dir):
    from os import listdir
    from os.path import isfile, join
    files = [f for f in listdir(text_dir) if isfile(join(text_dir, f))]
    ll = []
    for file in files:
        with open(fts_item_local_text_dir + file, 'r') as fin:
            ll.extend(fin.readlines())
    print('text file lines num:', len(ll))
    m = {}
    # print(ll[0:10])
    for line in ll:
        k, v = line.split(chr(1))
        ll = v.split(chr(4))
        for ele in ll:
            fts_name, fts_value = ele.split(chr(2))
            trim_v = str.rstrip(fts_value, '\n')
            if k in m:
                m[k][fts_name] = trim_v
            else:
                m[k] = {fts_name: trim_v}
    return m







if __name__ == '__main__':
    code_file = deploy_dir + 'inference.py'
    # sh.cd(deploy_dir)
    os.system('rm -rf %s' % deploy_tmp_dir)
    os.system('mkdir %s' % deploy_tmp_dir)
    os.system('mkdir %s' % deploy_code_dir)
    os.system('mkdir %s' % deploy_data_dir)
    os.system('mkdir %s' % fts_item_local_text_dir)
    os.system('cp %s %s'%(code_file,deploy_code_dir ))

    s3_cli = boto3.client('s3')
    sm_sess = sagemaker.Session()
    print('aws region:', sm_sess.boto_region_name)
    sm_cli = boto3.client('sagemaker')
    role = get_execution_role()
    print('role:', role)

    # download files
    # os.system('aws s3 cp --recursive %s %s' % (s3_model, deploy_tmp_dir))
    os.system('cp -r  %s %s' % (model_local, deploy_tmp_dir))
    os.system('aws s3 cp --recursive %s %s' % (fts_item_s3_text_dir, fts_item_local_text_dir))
    item_fts_dict = convert_text2pkl(fts_item_local_text_dir)
    with open(fts_item_pickle, 'wb') as fout:
        import pickle
        pickle.dump(item_fts_dict, fout)

    # tar
    tar_file = deploy_tmp_dir + tar_name
    os.system('tar -czvf  %s  %s' % (tar_file, deploy_tmp_dir))
    # upload
    s3_model_online_tar_file = s3_model_online + tar_name
    os.system('aws s3 cp %s %s' % (tar_file, s3_model_online_tar_file))
