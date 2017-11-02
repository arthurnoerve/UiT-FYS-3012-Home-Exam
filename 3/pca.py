import scipy.io as sio
import numpy.linalg as la
import numpy as np

import matplotlib.pyplot as plt

from alpha import *



dataX = sio.loadmat('Xtr_digits.mat')
X = dataX['Xtr_digits'].T


A,x = pca(X)

np.save("pca",x)
np.save("A",A)


'''
dataX = sio.loadmat('Xtr_digits_larger.mat')
X = dataX['Xtr_digits_larger'].T

A,x = pca(X)

np.save("pca_larger",x)
np.save("A_larger",A)
'''
