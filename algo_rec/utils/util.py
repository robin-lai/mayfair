import math
import os
import datetime
import json

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    c = math.ceil(len(lst) / n)
    for i in range(0, len(lst), c):
        yield lst[i:i + c]

def get_job_status_json(s3_file, local_file):
    os.system('rm %s' % local_file)
    os.system('aws s3 cp %s %s' % (s3_file, local_file))
    with open(local_file, 'r') as fin:
        js = dict(json.load(fin))
    return js

def put_job_file2s3(js, s3_file, local_file):
    with open(local_file, 'w') as fout:
        json.dump(js, fout)
    os.system('aws s3 cp %s %s' % (local_file, s3_file))

def pre_job_is_ready(job_name, pre_ds):
    js = get_job_status_json()
    if job_name == 'deploy_pkg':
        job = list(js['train'])
        job.sort(key=lambda x: x['idx'])
        lst_info = job[-1]
        if lst_info['status'] == 'ok' and lst_info['newest_ds'] == pre_ds:
            return True
        else:
            return False

import boto3
from botocore.exceptions import ClientError

def check_s3_file_exists(bucket_name, file_key):
    s3 = boto3.client('s3')
    print(f"{file_key} exists in {bucket_name}.")
    try:
        # Try to retrieve metadata of the file
        s3.head_object(Bucket=bucket_name, Key=file_key)
        return True  # File exists
    except ClientError as e:
        # If file does not exist, head_object will raise an exception
        if e.response['Error']['Code'] == '404':
            return False  # File does not exist
        else:
            raise  # Other errors

def add_job_monitor(job_name, dd):
    s3_file = 's3://warehouse-algo/rec/job_monitor.json'
    local_file = '/home/sagemaker-user/mayfair/algo_rec/job_monitor.json'
    js = get_job_status_json(s3_file, local_file)
    idx = len(js[job_name])
    pre_ds = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    today = datetime.date.today().strftime('%Y%m%d')
    d = {"idx": idx+1, "status":"ok","today":today, "newest_ds": pre_ds}
    d.update(dd)
    js[job_name].append(d)
    put_job_file2s3(js, s3_file, local_file)
    print('add %s status done'%job_name, d)