import tensorflow as tf
import numpy as np

tf.compat.v1.enable_eager_execution()
# Create a TensorFlow tensor
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

# Convert the tensor to a NumPy array
numpy_array = tensor.numpy()

# Save the NumPy array to a file
np.save('tensor_data.npy', numpy_array)

# To load the NumPy array back
loaded_array = np.load('tensor_data.npy')
print(loaded_array)