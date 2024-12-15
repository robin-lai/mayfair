import multiprocessing
import time
if __name__ == '__main__':

    manager = multiprocessing.Manager()
    manager_dict = manager.dict()
    manager_dict[1] = manager.dict()
    manager_dict[2] = manager.dict()
    manager_dict[3] = manager.dict()
    manager_dict[1]["ctr"] = manager.list([i for i in range(1000000)])
    manager_dict[1]["cvr"] = manager.list([i for i in range(1000000)])
    manager_dict[1]["ctcvr"] = manager.list([i for i in range(1000000)])
    manager_dict[1]["id"] = manager.list([i for i in range(1000000)])
    manager_dict[2]["ctr"] = manager.list([i for i in range(1000000)])
    manager_dict[2]["cvr"] = manager.list([i for i in range(1000000)])
    manager_dict[2]["ctcvr"] = manager.list([i for i in range(1000000)])
    manager_dict[2]["id"] = manager.list([i for i in range(1000000)])
    manager_dict[3]["ctr"] = manager.list([i for i in range(1000000)])
    manager_dict[3]["cvr"] = manager.list([i for i in range(1000000)])
    manager_dict[3]["ctcvr"] = manager.list([i for i in range(1000000)])
    manager_dict[3]["id"] = manager.list([i for i in range(1000000)])
    dd = {}
    dd.update(manager_dict)
    # dd2 = {}
    # for k, v in dd.items():
    #     st = time.time()
    #     for i, j in v.items():
    #         if i in dd2:
    #             dd2[i].extend(j)
    #         else:
    #             dd2[i] = j
    #     ed = time.time()
    #     print('%s cost:%s' % (str(k), str(ed - st)))

    # print('*' * 80)
    # dd3 = {}
    # for k, v in dd.items():
    #     st = time.time()
    #     for i, j in v.items():
    #         tmp_l = list(j)
    #         if i in dd3:
    #             dd3[i].extend(tmp_l)
    #         else:
    #             dd3[i] = j
    #     ed = time.time()
    #     print('%s cost:%s' % (str(k), str(ed - st)))

    print('*' * 80)
    dd4 = {}
    for k, v in dd.items():
        st = time.time()
        tmp_d = {}
        tmp_d.update(v)
        ed = time.time()
        print('cost1:', ed-st)
        for i, j in tmp_d.items():
            st = time.time()
            tmp_l = []
            print(type(j))
            # for e in j
            # tmp_l.extend(j)
            ed = time.time()
            print('cost2:', ed-st)
            st = time.time()
            if i in dd4:
                dd4[i].extend(tmp_l)
            else:
                dd4[i] = j
            ed = time.time()
            print('cost3:', ed-st)
        # ed = time.time()
        # print('%s cost:%s' % (str(k), str(ed - st)))