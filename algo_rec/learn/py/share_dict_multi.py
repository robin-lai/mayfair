import multiprocessing
import time

num = 1000000
def worker(proc, d):
    st = time.time()
    for i in range(num):
        d['k' + str(i)]
    ed = time.time()
    print('cost:',proc, ed-st)

        # print(f"Process {multiprocessing.current_process().name} - Key: {key}, Value: {shared_dict[key]}")


if __name__ == "__main__":
    # 创建一个只读字典
    data = {str('k' + str(i)):{"1":0.1, "2":0.2} for i in range(num)}

    # 使用 Manager 来创建字典
    st = time.time()
    manager = multiprocessing.Manager()
    shared_dict = manager.dict(data)
    ed = time.time()
    print('cost gen dict:', ed - st)
    st = time.time()

    worker(0, shared_dict)
    ed = time.time()
    print('1 proc cost:', ed-st)

    processes = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i, shared_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    ed = time.time()
    print('cost multi read cost:', ed-st)
