import multiprocessing
import time
from pympler import asizeof

num = 1000000
def worker(proc,d):
    """worker function"""
    def fun():
        for i in range(num):
            d["k" + str(i)]

    fun()


def main(d):
    proc_list = [multiprocessing.Process(target=worker, args=(proc,d)) for proc in range(2)]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]



if __name__ == "__main__":
    st = time.time()
    d2 = {"k"+str(i):{"k2"+str(i):i} for i in range(num)}
    print('size of mem [M]', asizeof.asizeof(d2) / 1048576)  #
    main(d2)
    ed = time.time()
    print('cost:', ed-st)
