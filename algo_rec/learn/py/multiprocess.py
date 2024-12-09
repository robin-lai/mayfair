import multiprocessing

d = {"1":2}

def worker(procnum, return_dict):
    """worker function"""
    print(str(procnum) + " represent!")
    return_dict[procnum]["ctr"] = procnum
    print(d["1"])
    return_dict[procnum]["2"] = d["1"]


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(5):
        return_dict[i] = manager.dict()
        p = multiprocessing.Process(target=worker, args=(i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    for k, v in return_dict.items():
        for i, j in v.items():
            print(i,j)
            print(k, dict(v))
    # print(dict(return_dict))
    # print(return_dict.values())