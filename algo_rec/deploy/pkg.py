
import boto3
import sagemaker
from sagemaker import get_execution_role
import sh
from constant import *

if __name__ == '__main__':
    deploy_dir =  '/home/sagemaker-user/mayfair/algo_rec/deploy/'
    deploy_tmp_dir = deploy_dir + 'tmp/'
    deploy_code_dir = deploy_tmp_dir + 'code/'
    code_file = deploy_dir + 'inference.py'
    sh.cd(deploy_dir)
    sh.mkdir(deploy_tmp_dir)
    sh.mkdir(deploy_code_dir)
    sh.cp(code_file, deploy_code_dir)

    s3_cli = boto3.client('s3')
    sm_sess = sagemaker.Session()
    print('aws region:', sm_sess.boto_region_name)
    sm_cli = boto3.client('sagemaker')
    role = get_execution_role()
    print('role:', role)

    # download files
    BUCKET = 'warehouse-algo'
    s3_cli.download_file(Bucket=BUCKET, Key=s3_model, Filename=deploy_tmp_dir)
    # tar
    tar_file = deploy_tmp_dir + tar_name
    sh.tar("czvf", tar_file, deploy_tmp_dir)
    # upload
    s3_model_online_tar_file = s3_model_online + tar_file
    s3_cli.upload_file(tar_file, BUCKET, s3_model_online)
