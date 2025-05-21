import os
os.system('git pull')
os.system('find algo_rec -type f -name "*.py" | tar -czvf algo_rec.tar.gz -T -')
os.system('aws s3 cp algo_rec.tar.gz s3://warehouse-algo/rec/code/algo_rec.tar.gz')

os.system('find supply_chain -type f -name "*.py" | tar -czvf supply_chain.tar.gz -T -')
os.system('aws s3 cp supply_chain.tar.gz s3://warehouse-algo/rec/code/supply_chain.tar.gz')
