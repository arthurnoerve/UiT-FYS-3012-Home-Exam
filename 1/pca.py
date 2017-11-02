import scipy.io as sio
import numpy.linalg as la
import numpy as np

import matplotlib.pyplot as plt

from alpha import *



# Load data from Matlab files
# 1420 samples with 4096 features each
data = sio.loadmat('seal_data.mat')
X = data['X']


A,x = pca(X)

np.save("seal_pca",x)
np.save("A",A)
