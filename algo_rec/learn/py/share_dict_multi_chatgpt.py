import multiprocessing
import json
from multiprocessing import shared_memory
import time


def worker(proc_id, shm_name, size):
    """子进程任务，从共享内存中读取嵌套字典并访问数据。"""
    # 连接到共享内存
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    try:
        # 读取共享内存的数据
        shared_bytes = existing_shm.buf[:size]
        shared_data = json.loads(shared_bytes.tobytes().decode())

        # 测试访问嵌套字典
        start = time.time()
        for i in range(1000):  # 模拟访问操作
            _ = shared_data[f'k{i}']['1']
        end = time.time()
        print(f"Process {proc_id} - Access time: {end - start:.4f} seconds")
    finally:
        # 关闭共享内存
        existing_shm.close()


if __name__ == "__main__":
    # 创建嵌套字典
    data = {f'k{i}': {"1": 0.1, "2": 0.2} for i in range(100000)}  # 示例数据
    serialized_data = json.dumps(data).encode()  # 序列化为 JSON

    # 创建共享内存
    shm = shared_memory.SharedMemory(create=True, size=len(serialized_data))
    shm.buf[:len(serialized_data)] = serialized_data  # 写入数据到共享内存

    print("Shared memory created. Name:", shm.name)

    # 启动子进程读取共享内存
    processes = []
    num_processes = 4  # 启动的子进程数量
    for proc_id in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(proc_id, shm.name, len(serialized_data)))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 清理共享内存
    shm.close()
    shm.unlink()
    print("Shared memory cleaned up.")
