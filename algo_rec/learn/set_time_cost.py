import time
def set_common(set1, set2, n):
    st = time.time()
    for i in range(n):
        common_set = set1 & set2
    ed = time.time()
    print('cost:', str(ed-st))



if __name__ == '__main__':
    set1 = set([str(i) for i in range(100)])
    set2 = set([str(i) for i in range(50,150,1)])
    n = 1000*10000
    set_common(set1, set2,n)