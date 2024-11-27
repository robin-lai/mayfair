import tensorflow as tf
import numpy as np
tensor1 = tf.constant([[1, 2, 3], [4, 5, 6]])
tensor2 = tf.constant([[1, 2, 3], [4, 5, 6]])
tensor3 = tf.constant([[1, 2, 3], [4, 5, 6]])
ll = [tensor2, tensor1, tensor3]


def get_tensor():
    for i in range(0,3):
        yield ll[i]

gen_it = get_tensor()

ret = []
for e in gen_it:
    ret.append(e.numpy().tolist())
import  pickle
with open('./list.pkl', 'wb') as fout:
    pickle.dump(ret, fout)

# print(gen_it.numpy())
