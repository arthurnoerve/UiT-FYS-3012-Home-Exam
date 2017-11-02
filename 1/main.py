import scipy.io as sio
import numpy.linalg as la
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from alpha import *
import os
from scipy.signal import savgol_filter
import scipy


# Load data from Matlab files
# 1420 samples with 4096 features each
data = sio.loadmat('seal_data.mat')
X = data['X']
Y = data['y']


np.seterr(over='ignore')

def scatter(data,**kwargs):
    w = kwargs.get("w",False)
    color = kwargs.get("c",False)

    if 'c' in kwargs:
        plt.scatter(*zip(*data),c=color,s=2)
    else:
        plt.scatter(*zip(*data),s=2)

    if 'w' in kwargs:
        xs = np.arange(-10,60)
        plt.plot(-w[0]/w[2]-w[1]/w[2]*xs)
    plt.show()



R = 0.7
V = 0.5
X_train, X_valid, X_test = split_data(X,R,V)
Y_train, Y_valid, Y_test = split_data(Y,R,V)




# 2LP =======================================================================================
def run_net(name, x,y):
    W1 = np.load(name+"_1.npy")
    W2 = np.load(name+"_2.npy")

    net = twolp(W1,W2,f,df)
    hit_2lp = twolp_test(net,x,y)

    return hit_2lp


# Define net
f = lambda x: max(0,x)
df = lambda x: 0 if x<=0 else 1
rho = 0.000001
mom = 0.1
name = "W_100neuron"

'''
W1,W2,cost = twolp_generate(X_train, Y_train, 100,f,df, rho, momentum=mom, iterations=2000,load=name)

np.save(name+"_1",W1)
np.save(name+"_2",W2)
cost_file = name+"_cost.npy"
cost = np.concatenate((np.load(cost_file), cost))if os.path.isfile(cost_file) else cost
np.save(cost_file,cost)
'''

'''
# plot cost
cost_file = name+"_cost.npy"
cost = np.load(cost_file)

smooth_cost = savgol_filter(cost, 99, 2)
plt.plot(cost,c='blue')
plt.plot(smooth_cost,c='red')
plt.show()
'''


'''
#transform data
A = np.load("A.npy")
a = A[:,:2]
X_train = np.dot(a.transpose(),X_train.transpose()).T
X_test = np.dot(a.transpose(),X_test.transpose()).T
'''

#hit test
hit_net = run_net(name,X_test,Y_test)
print(hit_net)









# SVM =======================================================================================


#a,b = linear_smo(X_train,Y_train)
#hit_smo = smo_test(a,b,X_train,Y_train,X_test,Y_test)
#print(hit_smo)

# Only 2d:
#plot_svm_train(X_train, Y_train, alphas, bes)




def test(r, data, labels):
    X_train, X_valid, X_test = split_data(data,r,0.5)
    Y_train, Y_valid, Y_test = split_data(labels,r,0.5)

    def f(x): return linear_test(widrow_hoff(X_train,Y_train,np.asscalar(x))[0], X_valid,Y_valid)
    best_rho = scipy.optimize.minimize(lambda x: -f(x), 0, bounds=((0,1),),options={'disp': False}).x
    best_rho = np.asscalar(best_rho)
    print("BEST_RHO: "+str(best_rho))

    (w_lms, c_lms) = widrow_hoff(X_train,Y_train,best_rho)
    hit_lms = linear_test(w_lms,X_test,Y_test)

    w_ls = least_squares(X_train,Y_train)
    hit_ls = linear_test(w_ls,X_test,Y_test)

    w_bay = bayesian(X_train,Y_train)
    hit_bay = linear_test(w_bay,X_test,Y_test)
    '''
    a,b = linear_smo(X_train,Y_train)
    hit_smv = smo_test(a,b,X_train,Y_train,X_test,Y_test)
    '''

    print("LMS: "+str(hit_lms))
    print("LS: "+str(hit_ls))
    print("BAY: "+str(hit_bay))
    #print("SVM: "+str(hit_smv))

    return  ( hit_lms,hit_ls,hit_bay )


def multi_test(r,n,data,labels):

    lms = np.zeros(n)
    ls = np.zeros(n)
    bay = np.zeros(n)
    for i in range(n):
        #shuffle data for each run
        ind = np.random.permutation(len(data))
        d = data[ind]
        l = labels[ind]
        print("Run " + str(i))
        t = test(r,d,l)

        lms[i] = t[0]
        ls[i] = t[1]
        bay[i] = t[2]


    return lms,ls,bay


#lms,ls,bay = multi_test(0.6,10,X,Y)


#plt.plot(lms)
#plt.plot(ls)
#plt.plot(bay)
#plt.legend(["LMS","LS","BAYES"])
#plt.show()



# DIM MULTI TEST  =======================================================================================


'''
A = np.load("A.npy")
a = A[:,:2]
X = np.dot(a.transpose(),X.transpose()).T

rs = [0.4, 0.6, 0.8]
v = 0.5

#rhos = {}

for r in rs:
    print("SIZE:" + str(r))

    N = 10
    hits = np.zeros(N)
    #rho = np.zeros(N)
    for i in range(N):

        #shuffle data for each run
        ind = np.random.permutation(len(X))
        x = X[ind]
        y = Y[ind]

        #split data
        d, X_valid, X_test = split_data(x,r,v)
        l, Y_valid, Y_test = split_data(y,r,v)

        #def f(x): return linear_test(widrow_hoff(d,l,np.asscalar(x))[0], X_valid,Y_valid)
        #best_rho = scipy.optimize.minimize(lambda x: -f(x), 0, bounds=((0,1),),options={'disp': False}).x
        #best_rho = np.asscalar(best_rho)
        #rho[i] = best_rho
        #print("BEST_RHO: "+str(best_rho))

        #w = least_squares(d,l)
        #w = bayesian(d,l)
        #w,_ = widrow_hoff(d,l,best_rho)
        a,b = linear_smo(d,l)


        #Test accuracy
        #hit = linear_test(w,X_test,Y_test)
        hit = smo_test(a,b,d,l,X_test,Y_test)
        hits[i] = hit


        print("Run " + str(i) + " - " + str(hit))

    #rhos[str(r)] = rho.mean()
    #calculate mean and variance
    mean = hits.mean()
    var = hits.var()
    print("Mean: " + str(mean))
    print("Var: " + str(var))

    #plot gaussian
    sigma = np.sqrt(var)
    xs = np.linspace(mean - 5*sigma, mean + 5*sigma, 100)
    plt.plot( xs, mlab.normpdf(xs, mean, sigma) )



plt.legend(["40%","60%","80%"])
#plt.legend(["40% - "+'%.2E' % rhos['0.4'], "60% - "+'%.2E' % rhos['0.6'], "80% - "+'%.2E' % rhos['0.8']])
plt.title("SVM classifier for different set sizes")
plt.savefig("dim_test/svm_2.png")
plt.show()
'''





# Stuff  =======================================================================================

A = np.load("A.npy")
def t(dim):
    dim = int(dim)

    a = A[:,:dim] # Transformation
    x = np.dot(a.transpose(),X.transpose()).T # Transformed data
    print("RUN FOR DIM: "+str(dim))
    return test(0.7,x,Y)


#fan(t,2,A.shape[1],n=100,plot=True,titles=["LMS","LS","BAY","SVM"])





'''
a = A[:,:2]

x = np.dot(a.transpose(),X.transpose()).T
x_train = np.dot(a.transpose(),X_train.transpose()).T
x_test = np.dot(a.transpose(),X_test.transpose()).T


w = bayesian(x_train,Y_train)


scatter(x,c=Y)
'''




# LEM  =======================================================================================

def construct_adj_mat(data, mode="fixed", **kwargs):
    X = np.array(data)
    t = kwargs.get("t",False)


    if mode == "fixed":
        epsilon = kwargs.get("epsilon",0.05)
        def is_connected(X,l,a,b):
            return l < epsilon
    elif mode == "knn":
        n = kwargs.get("n",5)
        def is_connected(X,l,x,y):
            ds = np.array([la.norm(x-z)**2 for z in X])
            i = ds.argsort()
            max_d = ds[i][n]
            return l < max_d


    if t:
        def get_weight(l):
            return np.exp(-l/t)
    else:
        def get_weight(l):
            return 1

    dim = len(X)
    A = np.zeros((dim,dim))

    for i in range(len(X)):
        x = X[i]
        print("Graphing for vector "+str(i))
        for j in range(len(X)):
            y = X[j]
            if i == j: continue
            l = la.norm(x-y)**2
            if is_connected(X,l,x,y):
                A[i,j] = get_weight(l)

    print("Graph done")

    np.save("LEM_A",A)

    # Solve problem
    D = np.sum(A, axis=0)
    L = D - A

    l,v = scipy.linalg.eig(L,D)
    print("Eigen problem done")

    idx = l.argsort()
    l = l[idx]
    v = v[:,idx]

    return v



#A = construct_adj_mat(X,mode="fixed",n=1)
