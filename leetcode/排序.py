

# native: 找到最小的， 找到次小的，。。。。。 n + n-1 + n-3 + ... + 1. n平方。
# 转化为剩余数中找最小，最小的放在相应的位置
# 找到最小，记录下idx, 最小值加入返回list, 最小idx值记录最大
def sort_native(arr):
    n = len(arr)
    ret = []
    for i in range(n):
        min = arr[i]
        idx = i
        for j in range(0, n):
            if arr[j] < min:
                min = arr[j]
                idx = j
        arr[idx] = 1000
        ret.append(min)
                # idx = j
        # arr[idx] = arr[i]
        # arr[i] = min
    print(ret)


if __name__ == '__main__':
    print(sort_native([2,4,5,1,8,9,10]))