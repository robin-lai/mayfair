from pyarrow import parquet
import argparse
from datetime import datetime,date, timedelta
import pickle

dd_sku_file = 'dd_sku.pkl'
dd_skc_file = 'dd_skc.pkl'
def model1(dd):
    for k, v in dd.items():
        print(v)
        v.sort(key=lambda x: x['sign_date'], reverse=True)
        print(v)




def main(args):
    dd_sku = {}
    if not args.debug:
        pt = parquet.read_table(args.dir_pt_sku).to_pylist()
        for t in pt:
            # t['sign_date_format'] = date(t['sign_date'])
            if t['sale_sku_id'] not in dd_sku:
                dd_sku[t['sale_sku_id']] = [t]
            else:
                dd_sku[t['sale_sku_id']].append(t)
        print(f"sku_num:{len(dd_sku)}")
        with open(dd_sku_file, 'wb') as fout:
            pickle.dump(dd_sku, fout)
    else:
        with open(dd_sku_file, 'rb') as fin:
            dd_sku = pickle.load(fin)
    model1({434453: dd_sku[434453]})





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='statistic',
        description='statistic',
        epilog='statistic-help')
    parser.add_argument('--ds', type=str,
                        default=(datetime.today() - timedelta(days=1)).strftime('%Y%m%d'))
    parser.add_argument('--dir_pt_sku', default='s3://warehouse-algo/supply_chain/sc_dto_num_sku_estimation/ds=%s')
    parser.add_argument('--dir_pt_skc', default='s3://warehouse-algo/supply_chain/sc_dto_num_skc_estimation/ds=%s')
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    args.dir_pt_sku = args.dir_pt_sku % args.ds
    args.dir_pt_skc = args.dir_pt_skc % args.ds
    main(args)
