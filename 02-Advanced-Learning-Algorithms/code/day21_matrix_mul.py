import numpy as np
import pandas as pd

arr1T = np.array([[1, -1, 0.1], [2, -2, 0.2]])
arr2 = np.array([[1, 2], [-1, -2], [0.1, 0.2]])
b = np.array([1, 2])
def dense(arr1T, arr2, b):
    z =  np.matmul(arr1T, arr2) + b
    aout = 1/(1+np.exp(-z))
    return aout

print(dense(arr1T, arr2, b))