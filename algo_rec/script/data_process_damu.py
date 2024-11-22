# -*- coding: utf-8 -*-
# from odps.udf import annotate
# from odps.udf import BaseUDTF

# @annotate("*->map<string,string>")
class seq_lookup_igraph_v2(object):
    def lookup_fts(self, lkp, id, k, action_type):
        if id in lkp and k in lkp[id]:
            lkp[id][k] = lkp[id][k] + 1
        elif id in lkp and k not in lkp[id]:
            lkp[id][k] = 1
        elif id not in lkp:
            lkp[id] = {}
            lkp[id][k] = 1
        else:
            print('unknown id')
    def evaluate(self, seq,ts,goods_id,goods_name, main_color, cate_id, cate_name, prop_seaon,prop_main_material,prop_style):
        if seq is None:
            return None
        lkp = {}

        def process_id(lkp,dist, scene, type, id, action_type):
            if dist <= 24 * 60 * 60:
                fts_prefix = type + '_in_seq_all_1d'
                if action_type == 'GOODS':
                    k = fts_prefix + '_clk'
                    self.lookup_fts(lkp, id, k, action_type)
                if action_type == 'ADD_WISH':
                    k = fts_prefix + '_wish'
                    self.lookup_fts(lkp, id, k, action_type)
                if action_type == 'ADD_BAG':
                    k = fts_prefix + '_bag'
                    self.lookup_fts(lkp, id, k, action_type)
                else:
                    print("unknown action type:",action_type)

        for ele in seq:
            if chr(3) in ele:
                seq_str, seq_type = ele.split(chr(3))
            else:
                seq_str = ele
            if seq_type == 'query':
                continue
            ele_t = seq_str.split(chr(2))
            goods_id_, ts_, action_type, goods_name_, main_color_, cate_id_, cate_name_, prop_seaon_,prop_main_material_,prop_style_ = \
            ele_t[0], ele_t[1], ele_t[2], ele_t[3], ele_t[4], ele_t[5], ele_t[6], ele_t[7], ele_t[8], ele_t[9]
            dist = float(ts_) - ts
            if goods_id != '-1' and goods_id == goods_id_:
                process_id(lkp, dist, 'all', 'goods_id', 'goods_id_' + goods_id, action_type)
        ret = {}
        for k, v in lkp.items():
            ll = []
            for k2, v2 in v.items():
                ll.append(k2 + ":" + str(v2))
            ll_str = ','.join(ll)
            ret[k] = ll_str
        return ret


if __name__ == '__main__':
    s = seq_lookup_igraph_v2()
    seq_list = ["1385301|17304604|GOODS|Denim Bodycon Dress with Belt|深蓝|781|Bodycon Dresses|Four Seasons|Cotton|Sexy", "1339101|17304604|GOODS|Open Back Slip Dress|杏色|527|Bodycon Dresses|Summer|Polyester|Minimalist"]
    s.evaluate(seq_list, 17308604, "1385301","Denim Bodycon Dress with Belt","深蓝", "781", "Bodycon Dresses", "Four Seasons", "Cotton", "Sexy" )
