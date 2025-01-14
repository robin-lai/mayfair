import multiprocessing

num = 100000
def worker(proc):
    """worker function"""
    for i in range(num):
        if i % 1000 == 0:
            print(f"proc {proc} v:{d["k" + str(i)]}")




if __name__ == "__main__":
    proc_list = [multiprocessing.Process(target=worker, args=(proc,)) for proc in range(5)]
    d = {"k"+str(i):{"k2"+str(i):i} for i in range(num)}
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]