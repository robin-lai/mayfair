import psutil

def get_available_memory():
    # 获取内存信息
    mem_info = psutil.virtual_memory()
    # 可用内存大小（以字节为单位）
    available_memory = mem_info.available
    return available_memory

if __name__ == "__main__":
    available_mem = get_available_memory()
    print(f"Available memory: {available_mem / (1024 * 1024):.2f} MB")
