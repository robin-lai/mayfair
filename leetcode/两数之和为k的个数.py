
def two_sum(arr, k):
    n = len(arr)
    ret = 0
    for i in range(0, n-1):
        for j in range(i+1,n):
            if arr[i] + arr[j] == k:
                # print(i, j)
                ret += 1
    return ret

def two_sum2(arr, k):
    n = len(arr)
    ret = 0
    cnt = {}
    for i in range(n):
        if k - arr[i] in cnt:
            ret += cnt[k-arr[i]]
        if arr[i] in cnt:
            cnt[arr[i]] = cnt[arr[i]] + 1
        else:
            cnt[arr[i]] = 1
    # print(cnt)
    return ret


if __name__ == '__main__':
    if 10 ==  two_sum2([1,1,1, 1, 1], 2):
        print('pass')
    else:
        print('not_pass')

