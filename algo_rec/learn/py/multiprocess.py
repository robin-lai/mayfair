import multiprocessing

d = {"1":2}

def worker(procnum, return_dict):
    """worker function"""
    print(str(procnum) + " represent!")
    # print(d["1"])
    for i in range(10):
        if 'ctr' in return_dict[procnum]:
            return_dict[procnum]['ctr'].extend([1,2,3,4,5])




if __name__ == "__main__":
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(2):
        return_dict[i] = manager.dict()
        return_dict[i]['ctr'] = manager.list([0,1,1,1])
        p = multiprocessing.Process(target=worker, args=(i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    for k, v in return_dict.items():
        for i, j in v.items():
            # print(i,j)
            print(k, dict(v))
            print(i, list(v['ctr']))
    # print(dict(return_dict))
    # print(return_dict.values())