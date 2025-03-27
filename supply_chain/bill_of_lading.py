import pandas as pd


rule_df = pd.read_csv('~/Downloads/ladder_goods_rule.csv', skip_blank_lines=True)
df = pd.read_csv('~/Downloads/ladder_price.csv', skip_blank_lines=True)
df['order_num'] = df['按14天销售周期算的备货量']

print(df.columns, df.shape)
print(rule_df.columns)

df_merge = df.merge(rule_df, how='left', left_on='goods_id', right_on='goods_id')
print(df_merge.columns)
# pd.Series.to_json()
import json
import numpy as np


def fun(x):
    try:
        d = x.to_dict()
        # print(d['rule_value'])
        # print(d['order_num'])
        rule_str = str(d['rule_value'])
        # if int(d['order_num']) is None:
        #     return '备货量为空', 0, 0, '无', None, None
        if 'changeValue' not in rule_str or 'rightValue' not in rule_str:
            return '无', 0, 0, '无', None, None
        print('rule_str', rule_str)
        rules = json.loads(rule_str)
        rules.sort(key=lambda x: float(x['leftValue']))
        for e in rules[1:]:
            # if float(d['order_num']) >= float(e['leftValue']) and float(d['order_num']) <= float(e['rightValue']):
            if float(d['order_num']) >= float(e['leftValue']):
                if 'rightValue' in e:
                    if float(d['order_num']) <= float(e['rightValue']):
                        return e, 1, 0, float(e['changeValue']), float(e['ruleValueType']), rules
                else:
                    return e, 1, 0, float(e['changeValue']), float(e['ruleValueType']), rules

        first_rule = rules[1]
        gap = int(first_rule['leftValue']) - int(d['order_num'])
        return first_rule, 0, -gap, float(first_rule['changeValue']), float(first_rule['ruleValueType']), rules
    except Exception as e:
        print(f'exception:{e}, {d['goods_id']}')
        return '备货量为空', 0, 0, '无', None, None


# print(df_merge[['rule_value']].head(100))
df_merge[['命中的阶梯', '是否命中阶梯', '要命中阶梯备货量gap', '优惠力度', '阶梯价类型', '所有阶梯']] = df_merge.apply(
    lambda x: fun(x), axis=1, result_type='expand')
# print(df_merge.head(1))
dd = {}
n = df_merge.shape[0] - 1
dd['goods总数'] = n
# dd['有阶梯价的goods总数'] = m
df_merge.to_csv('~/Downloads/ladder_price_bill_of_lading.csv')
