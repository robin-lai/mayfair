# encoding:utf-8

import pprint
from pyarrow import parquet
def main(lines):
    def swing(trig_itm, alph, user_debias=True):
        swing = {}
        user = list(item_bhv_user_list[trig_itm])
        u_num = len(user)
        if u_num < 2:
            return swing
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
                        swing[tgt_item] += score
                    else:
                        swing[tgt_item] = score
        return swing

    user_bhv_item_list = {}
    user_bhv_num = {}
    item_bhv_user_list = {}
    item_bhv_num = {}
    for line in lines:
        u, itm, clk = line[0], line[1], line[2]
        if u in user_bhv_item_list.keys():
            user_bhv_item_list[u].add(itm)
        else:
            user_bhv_item_list[u] = set(itm)
        if itm in item_bhv_user_list.keys():
            item_bhv_user_list[itm].add(u)
        else:
            item_bhv_user_list[itm] = set(u)
        # count
        if u in user_bhv_num.keys():
            user_bhv_num[u] += 1
        else:
            user_bhv_num[u] = 1
        if itm in item_bhv_num.keys():
            item_bhv_num[itm] += 1
        else:
            item_bhv_num[itm] = 1

    # print('user_bhv_item_list:', user_bhv_item_list)
    # print('item_bhv_user_list:', item_bhv_user_list)
    ret = {}
    for itm in item_bhv_user_list.keys():
        swing_rec = swing(itm, 1)
        ret[itm] = [(k, v) for k, v in swing_rec.items()]
    # pprint.pprint(ret, compact=True)
    return ret
    # pprint.pprint(swing('h',1))
    # pprint.pprint(swing('h',1, user_debias=False))



if __name__ == '__main__':
    # lines = [
    #     ("A", 'z', 1, 'cn'),
    #     ("A", 'p', 1, 'cn'),
    #     ("A", 't', 1, 'cn'),
    #     ("A", 'r', 1, 'cn'),
    #     ("A", 'h', 1, 'cn'),
    #     ("B", 'h', 1, 'cn'),
    #     ("B", 't', 1, 'cn'),
    #     ("B", 'r', 1, 'cn'),
    #     ("B", 'p', 1, 'cn'),
    #     ("C", 'h', 1, 'cn'),
    #     ("C", 'p', 1, 'cn'),
    #     ("C", 'y', 1, 'cn'),
    #     ("C", 'q', 1, 'cn'),
    #     ("D", 'h', 1, 'cn'),
    #     ("D", 'q', 1, 'cn'),
    #     ("E", 'h', 1, 'cn'),
    #     ("E", 'q', 1, 'cn'),
    #     ("E", 'o', 1, 'cn'),
    #     ("E", 'x', 1, 'cn'),
    # ]
    # lines = lines_t
    raw_file = 's3://algo-sg/rec/cn_rec_detail_recall_ui_relation/ds=20241118'
    pt = parquet.read_table(raw_file)
    m = {}
    for uuid, goods_id, clk_num, country_code in zip(pt['uuid'], pt['goods_id'],pt['clk_num'], pt['country_code']):
        country_code = country_code.as_py()
        t = (uuid.as_py(), goods_id.as_py(), clk_num.as_py())
        if country_code in m:
            m[country_code].append(t)
        else:
            m[country_code] = [t]
    ret = {}
    row_n = 30
    with open('./cn_rec_detail_recall_i2i_for_redis.txt', 'w') as fout:
        for k, v in m.items():
            ret[k] = main(v)
            for trig, tgt in ret[k].items():
                tgt.sort(key = lambda x: x[1], reverse=True)
                vs = []
                for ele in tgt:
                    row_n -= 1
                    vs.append(ele[0] + chr(4) + str(ele[1]))
                line = (k + chr(4) + trig + chr(1) + chr(2).join(vs) + '\n')
                fout.write(line)

    pass
