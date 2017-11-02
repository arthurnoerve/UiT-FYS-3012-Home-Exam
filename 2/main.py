import scipy.io as sio
import numpy.linalg as la
import numpy as np
import scipy
import matplotlib.pyplot as plt

from alpha import *

# Load data from Matlab files
# 135 samples with 62 features each
X = sio.loadmat('Xchl_tr.mat')['Xchl'].T        # [('Xchl', (62, 135), 'double')]
Y = sio.loadmat('ychl_tr.mat')['ychl_tr'][0]    # [('ychl_tr', (1, 135), 'double')]






def find_rho():
    rhos = np.linspace(0,0.0000000008,200)

    diff = np.zeros(len(rhos))
    for i in range(len(rhos)):
        r = rhos[i]
        y = loo_lms(X,Y,rho=r)

        diff[i] = sum( (y-Y)**2 )

    plt.plot(rhos,diff)
    plt.show()

    i = diff.argmin()
    return rhos[i]




# Leave out one test for funciton lms
def loo_lms(data,values,**kwargs):

    ys = np.zeros(len(data))
    if 'rho' not in kwargs:
        best_rhos = np.zeros(len(data))
    for i in range(len(data)):
        #leave out i
        dat_out = data[i]
        val_out = values[i]

        dat_rest = data[np.arange(len(data))!=i]
        val_rest = values[np.arange(len(values))!=i]

        def calc(w,x):
            return np.dot(augument(x),w)

        if 'rho' in kwargs:
            best_rho = kwargs.get('rho')
        else:
            rhos = np.linspace(0,0.0000000008,400)

            diff = np.zeros(len(rhos))
            for j in range(len(rhos)):
                r = rhos[j]
                W,_ = widrow_hoff(dat_rest,val_rest, r)
                S = (calc(W,dat_out) - val_out)**2
                diff[j] = S

            ind = diff.argmin()
            best_rho = rhos[ind]

            print("BEST_RHO: "+str(best_rho))


        w,_ = widrow_hoff(dat_rest,val_rest,best_rho)

        y = calc(w,dat_out)
        ys[i] = y
        best_rhos[i] = best_rho

    return ys,best_rhos

def loo_ls(data,values):

    ys = np.zeros(len(data))
    for i in range(len(data)):
        #leave out i
        dat_out = data[i]
        val_out = values[i]

        dat_rest = data[np.arange(len(data))!=i]
        val_rest = values[np.arange(len(values))!=i]

        def calc(w,x):
            return np.dot(augument(x),w)

        w = least_squares(dat_rest,val_rest)

        y = calc(w,dat_out)
        ys[i] = y

    return ys


#r = find_rho()
#print(r)


#y,best_rhos = loo_lms(X,Y)
y = loo_ls(X,Y)

plt.figure(0)
plt.plot(y, c="red")
plt.plot(Y,c="blue")

plt.legend(["LS Regression","Original data"])

#plt.figure(1)
#plt.plot(best_rhos)

plt.show()


#print(best_rhos.mean())
D = sum( (y-Y)**2 )
print(D)
