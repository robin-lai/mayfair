import math
import os
import datetime
import json

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    c = math.ceil(len(lst) / n)
    for i in range(0, len(lst), c):
        yield lst[i:i + c]

def add_job_monitor(job_name, dd):
    s3_file = 's3://warehouse-algo/rec/job_monitor.json'
    local_file = '/home/sagemaker-user/mayfair/algo_rec/job_monitor.json'
    os.system('rm %s' % local_file)
    os.system('aws s3 cp %s %s' % (s3_file, local_file))
    with open(local_file, 'r') as fin:
        js = dict(json.load(fin))
    idx = len(js[job_name])
    pre_ds = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    today = datetime.date.today().strftime('%Y%m%d')
    d = {"idx": idx+1, "status":"ok","today":today, "newest_ds": pre_ds}
    d.update(dd)
    js[job_name].append(d)
    with open(local_file, 'w') as fout:
        json.dump(js, fout)
    os.system('aws s3 cp %s %s' % (local_file, s3_file))
    print('add %s status done'%job_name, d)