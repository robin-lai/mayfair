
import json

j = [{"1":2}]
list = [{'id': 123, 'data': 'qwerty', 'indices': [1,10]}, {'id': 345, 'data': 'mnbvc', 'indices': [2,11]}]

with open('./auc.json', 'w') as fout:
    json.dump(j, fout)
