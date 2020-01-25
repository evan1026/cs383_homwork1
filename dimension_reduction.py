from PIL import Image
import numpy as np
import os
from numpy import linalg as LA
import matplotlib.pyplot as plt


DIR = 'yalefaces'
SEPARATOR = '\\'
orig_data = []
std_data = []


def standardize(arr):
    mean = arr.mean()
    std = arr.std(ddof=1)
    new_arr = np.zeros(arr.size)
    i = 0
    for x in np.nditer(arr):
        new_arr[i] = (x - mean) / std
        i += 1
    return new_arr


def get_k_principal_components(data, k):
    cov = np.cov(data)
    d, v = LA.eig(cov)
    z = list(zip(d, v))
    z.sort()
    z.reverse()
    d, v = list(zip(*z))
    max_vectors = v[:2]
    return max_vectors


files = os.listdir(DIR)
for filename in files:
    if not filename.endswith('.txt'):
        im = Image.open(DIR + SEPARATOR + filename).resize((40, 40))
        arr = np.array(im).flatten()
        orig_data.append(arr)
        std_data.append(standardize(arr))

w = get_k_principal_components(std_data, 2)
matrix_w = np.hstack((w[0].reshape(154, 1),
                      w[1].reshape(154, 1)))

'''Z = np.matmul(matrix_w, std_data)
plt.scatter(Z[0], Z[1])
plt.show()'''





