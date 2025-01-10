import json
import pprint
import numpy as np

with open('./h5_response3.json', 'r') as fin:
    js = json.load(fin)

info = []
for idx, e in enumerate(js['data']['informationBOList']):
    track = json.loads(e['itemTrack'])

    tmp = {'rk':idx+1, 'goodsId': e['goodsId'], 'score':track['otherAttribute'], 's':track['s']}
    info.append(tmp)
pprint.pprint(info)
d = {}
for e in info:
    for s in e['s'].split(','):
        if s in d:
            d[s].append([e["goodsId"], e['rk']])
        else:
            d[s] = [[e["goodsId"], e['rk']]]


ll = []
for k, v in d.items():
    ll.append({"s": k, "n":len(v), 'ratio': len(v)/20, 'pos':np.mean([e[1] for e in v])})
ll.sort(key=lambda x: x.get('n'),reverse=True)
pprint.pprint(ll)