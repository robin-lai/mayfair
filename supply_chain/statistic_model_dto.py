from pyarrow import parquet
import argparse
from datetime import datetime, date, timedelta
import pickle

dd_sku_file = 'dd_sku.pkl'
dd_skc_file = 'dd_skc.pkl'
cols = ['goods_id',
        'skc_id',
        'brand',
        'site_code',
        'sign_date',
        'sign_sku_qty',
        'dto_sku_qty',
        'quality_dto_sku_qty',
        'size_dto_sku_qty',
        'layer_tag']

def process_data(debug, in_file, case_id,out_file, flag):
    dd = {}
    if not debug:
        pt = parquet.read_table(in_file).to_pylist()
        for t in pt:
            # t['sign_date_format'] = date(t['sign_date'])
            if t['sign_date'] is None:
                print('sign_date is None')
                print(t)
                continue
            if t['layer_tag'] == 'C':
                continue
            if flag == 'sku':
                tt = [t['sale_sku_id']]
            else:
                tt = [-1]
            for col in cols:
                tt.append(t[col])

            if tt[0] not in dd:
                dd[tt[0]] = [tt]
            else:
                dd[tt[0]].append(tt)
        print(f"dd_num:{len(dd)}")
        with open(out_file, 'wb') as fout:
            pickle.dump(dd, fout)
    else:
        with open(out_file, 'rb') as fin:
            dd = pickle.load(fin)
    model1({case_id: dd[case_id]})

def model1(dd):
    for k, v in dd.items():
        v.sort(key=lambda x: x[5], reverse=True)
        print([(t[0], t[5], t[6], t[7]) for t in v])


def main(args):
    if args.flag == 'sku':
        process_data(args.debug, args.dir_pt_sku, 434453, dd_sku_file, 'sku')
    elif args.flag == 'skc':
        process_data(args.debug, args.dir_pt_skc, 85192, dd_skc_file, 'skc')




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
    parser.add_argument('--flag', default='sku')
    args = parser.parse_args()
    args.dir_pt_sku = args.dir_pt_sku % args.ds
    args.dir_pt_skc = args.dir_pt_skc % args.ds
    main(args)
