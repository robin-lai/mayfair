import multiprocessing
import time




if __name__ == '__main__':
    manager = multiprocessing.Manager()
    st = time.time()
    ll = [i for i in range(1000000)]
    ed = time.time()
    print('cost0:', str(ed-st))
    manager_list = manager.list()
    st = time.time()
    for i in ll:
        manager_list.append(i)
    ed = time.time()
    print('cost1:', str(ed-st))

    st = time.time()
    tmp_list = []
    for i in ll:
        tmp_list.append(i)
    manager_list2 = manager.list(tmp_list)
    ed = time.time()
    print('cost2:', str(ed-st))

    st = time.time()
    manager_list3 = manager.list()
    tmp_list = []
    for i in ll:
        tmp_list.append(i)
    manager_list2.extend(tmp_list)
    ed = time.time()
    print('cost3:', str(ed - st))

    st = time.time()
    manager_dict3 = manager.dict()
    tmp_dict = {}
    for i in ll:
        tmp_dict[i] = i
    manager_dict3.update(tmp_dict)
    ed = time.time()
    print('cost4:', str(ed - st))


