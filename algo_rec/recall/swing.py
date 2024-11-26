# encoding:utf-8

import pprint
import argparse
import time
from pyarrow import parquet
import pickle
def process(lines):
    def swing(trig_itm, alph, user_debias=True):
        swing = {}
        user = list(item_bhv_user_list[trig_itm])
        u_num = len(user)
        if u_num < 2:
            return swing
        # print('common user num:', u_num)
        for i in range(0, u_num-1):
            for j in range(i + 1, u_num):
                # print('user a', user[i], 'user b', user[j])
                common_items = user_bhv_item_list[user[i]] & user_bhv_item_list[user[j]]
                common_items = common_items - set(trig_itm)
                for tgt_item in common_items:
                    if user_debias:
                        score = round((1 / user_bhv_num[user[i]]) * (1 / user_bhv_num[user[j]]) * (
                                    1 / (alph + (len(common_items)))), 4)
                    else:
                        score = round((1 / (alph + (len(common_items)))), 4)
                    if tgt_item in swing:
                        swing[tgt_item] = round(swing[tgt_item] +  score, 4)
                    else:
                        swing[tgt_item] = score
        return swing

    st = time.time()
    user_bhv_item_list = {}
    user_bhv_num = {}
    item_bhv_user_list = {}
    item_bhv_num = {}
    for line in lines:
        u, itm, clk = line[0], line[1], line[2]
        if u in user_bhv_item_list.keys():
            user_bhv_item_list[u].add(itm)
        else:
            user_bhv_item_list[u] = set([itm])
        if itm in item_bhv_user_list.keys():
            item_bhv_user_list[itm].add(u)
        else:
            item_bhv_user_list[itm] = set([u])
        # count
        if u in user_bhv_num.keys():
            user_bhv_num[u] += 1
        else:
            user_bhv_num[u] = 1
        if itm in item_bhv_num.keys():
            item_bhv_num[itm] += 1
        else:
            item_bhv_num[itm] = 1
    ed = time.time()
    print('data process for swing cost:',str(ed-st))
    print('data desc: user num %s, item num:%s'%(len(user_bhv_num.keys()), len(item_bhv_user_list.keys())))
    print('dump data into file')
    with open('./item_bhv_user_list.pkl', 'wb') as fout:
        pickle.dump(item_bhv_user_list, fout)
    with open('./user_bhv_item_list.pkl', 'wb') as fout:
        pickle.dump(user_bhv_item_list, fout)
    with open('./user_bhv_num.pkl', 'wb') as fout:
        pickle.dump(user_bhv_num, fout)
    with open('./item_bhv_num.pkl', 'wb') as fout:
        pickle.dump(item_bhv_num, fout)



    # print('user_bhv_item_list:', user_bhv_item_list)
    # print('item_bhv_user_list:', item_bhv_user_list)
    ret = {}
    c = 0
    st = time.time()
    hot_item_num = 0
    for itm in item_bhv_user_list.keys():
        if item_bhv_num[itm] > 2000:
            hot_item_num += 1
            continue
        c += 1
        swing_rec = swing(itm, 1)
        ret[itm] = [(k, v) for k, v in swing_rec.items()]
        if c % 100 == 0:
            ed = time.time()
            print('process 100 item cost:',str(ed-st))
    print('hot_item_num:', hot_item_num)
    print('dump swing result to file')
    with open('./swing_i2i_ret.pkl', 'wb') as fout:
        pickle.dump(ret, fout)
    # pprint.pprint(ret, compact=True)
    return ret
    # pprint.pprint(swing('h',1))
    # pprint.pprint(swing('h',1, user_debias=False))

def get_data_from_s3(raw_file):
    st = time.time()
    print('begin read parquet data from file:', raw_file)
    pt = parquet.read_table(raw_file)
    m = {}
    for uuid, goods_id, clk_num, country_code in zip(pt['uuid'], pt['goods_id'], pt['clk_num'], pt['country_code']):
        country_code = country_code.as_py()
        t = (uuid.as_py(), goods_id.as_py(), clk_num.as_py())
        if country_code in m:
            m[country_code].append(t)
        else:
            m[country_code] = [t]
    ed = time.time()
    print('end read file,cost:%s'%(str(ed - st)))
    print('data describe:', '*' * 50)
    for k, v in m.items():
        print('country:%s lines:%s' % (k, len(v)))
    return m

def get_mock_data():
    m = {'cn': [
        ("A", 'z', 1, 'cn'),
        ("A", 'p', 1, 'cn'),
        ("A", 't', 1, 'cn'),
        ("A", 'r', 1, 'cn'),
        ("A", 'h', 1, 'cn'),
        ("B", 'h', 1, 'cn'),
        ("B", 't', 1, 'cn'),
        ("B", 'r', 1, 'cn'),
        ("B", 'p', 1, 'cn'),
        ("C", 'h', 1, 'cn'),
        ("C", 'p', 1, 'cn'),
        ("C", 'y', 1, 'cn'),
        ("C", 'q', 1, 'cn'),
        ("D", 'h', 1, 'cn'),
        ("D", 'q', 1, 'cn'),
        ("E", 'h', 1, 'cn'),
        ("E", 'q', 1, 'cn'),
        ("E", 'o', 1, 'cn'),
        ("E", 'x', 1, 'cn'),
    ]}
    return m

def get_test_data():
    st = time.time()
    file = './cn_rec_detail_recall_ui_relation.txt'
    print('begin read parquet data from file:', file)
    m = {}
    ll = []
    with open(file, 'r') as fout:
        lines = fout.readlines()
        for line in lines:
            ll.append([e.strip('\n') for e in line.split(' ')])
    m['cn'] = ll
    ed = time.time()
    print('end read file,cost:%s'%(str(ed - st)))
    print('data describe:', '*' * 50)
    for k, v in m.items():
        print('country:%s lines:%s' % (k, len(v)))
    return m

def main(args):
    st = time.time()
    in_file = 's3://algo-sg/rec/cn_rec_detail_recall_ui_relation/ds=20241118'
    out_file = './cn_rec_detail_recall_i2i_for_redis_s3_20241118.txt'
    if args.flag == 's3':
        m = get_data_from_s3(in_file)
    elif args.flag == 'mock':
        m = get_mock_data()
    elif args.flag == 'sample':
        m = get_test_data()
    else:
        print('unknown flag:',args.flag)
    # print(m)

    ret = {}
    row_n = 30
    with open(out_file, 'w') as fout:
        for k, v in m.items():
            ret[k] = process(v)
            for trig, tgt in ret[k].items():
                tgt.sort(key = lambda x: x[1], reverse=True)
                vs = []
                for ele in tgt:
                    row_n -= 1
                    if row_n == 0:
                        break
                    vs.append(ele[0] + chr(4) + str(ele[1]))
                line = (k + chr(4) + trig + chr(1) + chr(2).join(vs) + '\n')
                fout.write(line)
    print('swing end cost:', str(time.time() - st))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='swing',
        description='swing-args',
        epilog='swing-help')
    parser.add_argument('--flag',default='mock')
    args = parser.parse_args()
    main(args)
