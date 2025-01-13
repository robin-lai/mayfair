import multiprocessing
from multiprocessing import shared_memory
import pickle
import time
read_num = 10000000
def worker(proc_id, shm_name, shm_size):
    """子进程读取共享内存中的嵌套字典并访问其内容。"""
    # 连接到现有共享内存
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    try:
        # 从共享内存中读取序列化数据
        serialized_data = bytes(existing_shm.buf[:shm_size])
        nested_dict = pickle.loads(serialized_data)  # 反序列化为嵌套字典

        # 模拟对嵌套字典的访问
        start = time.time()
        for i in range(read_num):  # 假设访问 1000 次
            _ = nested_dict[f'key_{i}']['inner_key']
        end = time.time()
        print(f"Process {proc_id} - Access time: {end - start:.4f} seconds")
    finally:
        # 关闭共享内存
        existing_shm.close()

if __name__ == "__main__":
    # 创建一个嵌套字典
    nested_dict = {f'key_{i}': {'inner_key': i, 'value': i * 2} for i in range(read_num)}
    start = time.time()
    for i in range(read_num):  # 假设访问 1000 次
        _ = nested_dict[f'key_{i}']['inner_key']
    end = time.time()
    print(f"Process only - Access time: {end - start:.4f} seconds")

    # 序列化嵌套字典
    serialized_data = pickle.dumps(nested_dict)
    shm_size = len(serialized_data)

    # 创建共享内存并写入序列化数据
    shm = shared_memory.SharedMemory(create=True, size=shm_size)
    shm.buf[:shm_size] = serialized_data  # 写入数据
    print(f"Shared memory created. Name: {shm.name}, Size: {shm_size} bytes")

    # 启动多个子进程读取共享内存
    processes = []
    num_processes = 4  # 启动 4 个子进程
    for proc_id in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(proc_id, shm.name, shm_size))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 清理共享内存
    shm.close()
    shm.unlink()
    print("Shared memory cleaned up.")
