from multiprocessing import Process
def f(*args):
    x = args[0]
    y = args[1]
    print('x', x)
    print('y', y)


if __name__ == '__main__':
    p = Process(target=f,args=[[1,2,3,4], [5,6,7,8]])
    p.start()
    p.join()