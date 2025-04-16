from pyarrow import parquet
import argparse
from datetime import datetime,date, timedelta

def main(args):
    dd = {}
    pt = parquet.read_table(args.dir_pt).to_pylist()
    for t in pt:
        if t['goods_id'] in dd:
            dd[t['goods_id']][t['rk']] = t
        else:
            dd[t['goods_id']] = {t['rk']: t}
    print(f"goods_id num {len(dd)}")

    # smooth flash sale
    for k, v in dd.items():
        v.sort(key=lambda x: x['rk'])




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='statistic',
        description='statistic',
        epilog='statistic-help')
    parser.add_argument('--ds', type=str,
                        default=(datetime.today() - timedelta(days=1)).strftime('%Y%m%d'))
    parser.add_argument('--dir_pt', default='s3://warehouse-algo/supply_chain/sc_sale_predict_statistic_model_skc_last_4w/ds=%s')
    args = parser.parse_args()
    args.dir_pt = args.dir_pt % args.ds
    main(args)
