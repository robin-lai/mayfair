
def convert_parquet2txt(raw_file, out_file):
    from pyarrow import parquet
    lines_t = parquet.read_table(raw_file).to_pylist()
    f = open('', 'w')
    for d in lines_t:
        line = ' '.join(str(d[x]) for x in d.keys())
        f.write(line + '\n')
    f.close()
# raw_file = 's3://algo-sg/rec/cn_rec_detail_recall_ui_relation/ds=20241119'
# out_file = './cn_rec_detail_recall_ui_relation.txt'
# convert_parquet2txt(raw_file, out_file)

import pickle

# with open('saved_dictionary.pkl', 'wb') as f:
#     pickle.dump(dictionary, f)
#
# with open('saved_dictionary.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)