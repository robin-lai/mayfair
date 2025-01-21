import pandas as pd

df = pd.read_csv('~/Downloads/percentile.csv')


fc_col = """
 %s = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="%s"),boundaries=[%s])
"""
emb_col = """ %s_emb = tf.feature_column.embedding_column(tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key="%s"),boundaries=[%s]), 4)"""

d = df.to_dict(orient='dict')
print(d)
ll = []
for k in d:
    if k == 'ds':continue
    name = k[3:]
    bin = d[k][0]
    set_bin = set()
    bin_filter = []
    for e in bin.split(','):
        if e not in set_bin:
            set_bin.add(e)
            v = round(float(e),5)
            bin_filter.append(str(v))
    print(emb_col % (name,name,','.join(bin_filter)))
    ll.append(name + '_emb')

print(ll)
