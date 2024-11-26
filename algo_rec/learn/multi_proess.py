from multiprocessing import Process
def f(x,y):
    print('x+y', x+y)
    return x+y

if __name__ == '__main__':
    p = Process(target=f,args=(1,2,))
    p.start()
    p.join()