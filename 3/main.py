import scipy.io as sio
import numpy.linalg as la
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from alpha import *
import os
from scipy.signal import savgol_filter
import scipy

from y2ascii import y2ascii

# Load data from Matlab files


def loadSmaller():
    dataX = sio.loadmat('Xtr_digits.mat')
    dataY = sio.loadmat('ytr_digits.mat')
    X = dataX['Xtr_digits'].T
    Y = dataY['ytr_digits'].T
    A = np.load("A.npy")

    return X,Y,A

def loadLarger():
    dataX = sio.loadmat('Xtr_digits_larger.mat')
    dataY = sio.loadmat('ytr_digits_larger.mat')
    X = dataX['Xtr_digits_larger'].T
    Y = dataY['ytr_digits_larger'].T
    A = np.load("A_larger.npy")

    return X,Y,A

X,Y,A = loadSmaller()
Xl,Yl,Al = loadLarger()


def scatter(data,**kwargs):
    w = kwargs.get("w",False)
    color = kwargs.get("c",False)

    if 'c' in kwargs:
        plt.scatter(*zip(*data),c=color,s=2)
    else:
        plt.scatter(*zip(*data),s=2)

    if 'w' in kwargs:
        xs = np.arange(2,15)
        plt.plot(xs,-(w[0]+w[1]*xs)/w[2])
    plt.show()






def test(r, data, labels, xl, yl):
    #X_train, X_valid, X_test = split_data(data,r,0.5)
    #Y_train, Y_valid, Y_test = split_data(labels,r,0.5)

    X_train = data
    Y_train = labels

    _, X_valid, X_test = split_data(xl,0,0.7)
    _, Y_valid, Y_test = split_data(yl,0,0.7)

    rhos = np.linspace(0,0.3,400)
    diff = np.zeros(len(rhos))
    for j in range(len(rhos)):
        r = rhos[j]
        W,_ = widrow_hoff(X_train,Y_train, r)
        S = linear_test(W, X_valid,Y_valid)
        diff[j] = S


    ind = diff.argmax()
    best_rho = rhos[ind]

    print("BEST_RHO: "+str(best_rho))

    (w_lms, c_lms) = widrow_hoff(X_train,Y_train,best_rho)
    hit_lms = linear_test(w_lms,X_test,Y_test)

    w_ls = least_squares(X_train,Y_train)
    hit_ls = linear_test(w_ls,X_test,Y_test)

    w_bay = bayesian(X_train,Y_train)
    hit_bay = linear_test(w_bay,X_test,Y_test)


    a,b = linear_smo(X_train,Y_train)
    hit_smv = smo_test(a,b,X_train,Y_train,X_test,Y_test)


    print("LMS: "+str(hit_lms))
    print("LS: "+str(hit_ls))
    print("BAY: "+str(hit_bay))
    print("SVM: "+str(hit_smv))

    return  ( hit_lms,hit_ls,hit_bay,hit_smv )




'''
a = A[:,:2]
x = np.dot(a.transpose(),X.transpose()).T.real

f = lambda y: "red" if y==-1 else "blue"
plt.scatter(*zip(*x), c=[f(y) for y in Y],s=10)
plt.savefig("pca_larger")
plt.show()
'''

'''
w,c = widrow_hoff(x,Y,min_rho)

print(w)
scatter(x,w)
'''


def t(dim):
    dim = int(dim)
    a = A[:,:dim] # Transformation
    al = Al[:,:dim] # Transformation
    x = np.dot(a.transpose(),X.transpose()).T.real # Transformed data
    xl = np.dot(al.transpose(),Xl.transpose()).T.real # Transformed data
    print("RUN FOR DIM: "+str(dim))
    return test(0.7,x,Y,xl,Yl)


#fan(t,2,A.shape[1],n =A.shape[1]-1,plot=True,titles=["LMS","LS","BAY","SVM"])

'''
_, X_valid, X_test = split_data(Xl,0,0.7)
_, Y_valid, Y_test = split_data(Yl,0,0.7)

a = A[:,:2]
al = Al[:,:2]
x = np.dot(a.transpose(),X.transpose()).T.real
xte = np.dot(al.transpose(),X_test.transpose()).T.real
xvl = np.dot(al.transpose(),X_valid.transpose()).T.real

#w_ls = least_squares(x,Y)

rhos = np.linspace(0,0.3,400)
diff = np.zeros(len(rhos))
for j in range(len(rhos)):
    r = rhos[j]
    W,_ = widrow_hoff(x,Y, r)
    S = linear_test(W, xvl,Y_valid)
    diff[j] = S


ind = diff.argmax()
best_rho = rhos[ind]

print("BEST_RHO: "+str(best_rho))

w_ls,_ = widrow_hoff(x,Y,best_rho)

f = lambda y: "red" if y==-1 else "blue"
scatter(xte,w=w_ls,c=[f(y) for y in Y_test])
'''

'''
a,b = linear_smo(x,Y)
hit_smv = smo_test(a,b,x,Y,xte,Y_test)


plot_svm_train(x, Y.reshape((len(Y),)), a, b)
'''






#a = A[:,:2]
#al = Al[:,:2]
#x = np.dot(a.transpose(),X.transpose()).T.real


rhos = np.linspace(0,0.3,400)
diff = np.zeros(len(rhos))
for j in range(len(rhos)):
    r = rhos[j]
    W,_ = widrow_hoff(Xl,Yl, r)
    S = linear_test(W, X,Y)
    diff[j] = S


ind = diff.argmax()
best_rho = rhos[ind]
print("BEST_RHO: "+str(best_rho))

w,_ = widrow_hoff(Xl,Yl,best_rho)


bits = sio.loadmat('Xte_digits_2017.mat')['Xte_digits_2017'].T

labels = [ linear_classify(w,augument(bit)) for bit in bits ]

print(y2ascii(labels))
