import numpy as np

def get_indices(value, numpy_array):
  '''
  Returns the indices in the numpy array which contains given value
  '''
  return np.argwhere(numpy_array == value)[0, 0]