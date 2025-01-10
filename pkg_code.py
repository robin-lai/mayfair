import os
os.system('cd algo_rec')
os.system('git pull')
os.system('cd ..')
os.system('tar -cvf algo_rec.tar.gz algo_rec')
os.system('aws s3 cp algo_rec.tar.gz s3://warehouse-algo/rec/code/algo_rec.tar.gz')