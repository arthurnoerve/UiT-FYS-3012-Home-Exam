import scipy.io as sio
import numpy.linalg as la
import numpy as np

import matplotlib.pyplot as plt


from svm_util import *

# Analysis ==============================================
# Reduce dimension and plot

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def augument(input):
    return np.insert(input, 0,1)

def fan(f,min,max, **kwargs):
    n = kwargs.get('n', 100)
    should_plot = kwargs.get('plot', False)
    titles = kwargs.get('titles', False)
    mode = kwargs.get('mode', 'min')
    xs = np.linspace(min,max, num=n )
    fs = []
    for x in xs:
            fs.append(f(x))

    if should_plot:
        plt.plot(xs,fs)
        if titles:
            plt.legend(titles)
        plt.show()

    if mode == "min": m = np.argmin(fs)
    elif mode == "max": m = np.argmax(fs)
    return (xs[m], fs[m] ) # return argument value and the function value

def linear_classify(w,x):
    i = np.dot(x,w)
    if i > 0:
        return 1
    elif i < 0:
        return -1
    else:
        return 0

def linear_test(w,testing,labels):

    hit = 0
    miss = 0

    for i in range(0,len(testing)):
        x = augument(testing[i])
        y = labels[i]

        yh = linear_classify(w,x)

        if y == yh:
            hit = hit + 1
        else:
            miss = miss + 1

    l = len(testing)
    summary = (
        hit/l,
        miss/l
    )

    return summary[0]

def split_data(data,p,v):

    s = int(len(data)*p)
    train = data[:s]

    rest = data[s:]
    t = int(len(rest)*v)
    valid = rest[:t]
    test  = rest[t:]
    return ( train,valid,test )




def pca(data, **kwargs):
    x = np.array(data)
    cutoff = kwargs.get('cutoff', False)

    u = x.mean(axis=0)
    h = np.ones((len(x),1))
    b = x - np.outer( h,u )

    print("Computing correlation matrix")
    Rx = np.dot(x.transpose(),x) / len(x)

    print("Computing eigenstuff")
    l,v = la.eig(Rx)

    # Sort by size
    print("Sorting by size of eigenvalues")
    idx = l.argsort()[::-1]
    l = l[idx]
    v = v[:,idx]

    p = np.cumsum(l/sum(l))

    plt.plot(p)
    plt.savefig("cumsum_eigen_pca")

    cutoff_index = np.argmax(p>0.99)
    print("Cutoff index: " + str(cutoff_index))

    print("Calculating transformation matrix")
    if cutoff:
        A = v[:,:cutoff]
    else:
        A = v[:,:cutoff_index]

    print(A.shape)

    y = np.dot(A.transpose(),x.transpose())
    print(y.shape)

    return (A, y.transpose())


# Classify ==============================================

# LS
def least_squares(training, labels):
    x = np.hstack((np.ones((len(training),1)),training))
    y = np.array(labels)

    # Form sample correlation
    xtx = np.dot(x.transpose(),x)
    #xtx = sum( [ np.outer(x[i],x[i]) for i in range(0,len(training)) ] )
    # Form sample training-label cross-correlation
    a = np.dot(x.transpose(),y)
    #a = sum( [ x[i].T*y[i] for i in range(0,len(training)) ] )

    try:
        xtxi = la.inv(xtx)
    except:
        return np.zeros(len(training[0])+1)

    w = np.dot(xtxi,a)
    return w

# LMS
def widrow_hoff(training, labels, rho):
    w = np.zeros( len(training[0])+1 )
    cost = np.zeros( len(training) )

    for i in range(0,len(training)):
        x = augument(training[i])
        y = labels[i]
        e = y - np.dot(x,w)

        cost[i] = e**2
        w = w + rho*e*x.transpose()


    return (w,cost)



def bayesian(training, labels):
    ones = np.nonzero(labels==1)[0]
    twos = np.nonzero(labels==-1)[0]

    training1 = training[ones]
    labels1 = labels[ones]
    training2 = training[twos]
    labels2 = labels[twos]


    p1 = len(training1)/len(labels)
    p2 = len(training2)/len(labels)

    mu1 = training1.mean(axis=0)
    mu2 = training2.mean(axis=0)

    b1 = training1 - np.outer( np.ones((len(training1))), mu1 )
    b2 = training2 - np.outer( np.ones((len(training2))), mu2 )

    sig1 = np.cov(b1, rowvar=False)
    sig2 = np.cov(b2, rowvar=False)
    S = ((len(b1)-1)*sig1+(len(b2)-1)*sig2)/(len(training)-2)


    try:
        Si = la.inv(S)
    except:
        return np.zeros(len(training[0])+1)
    w = np.dot( Si,(mu1-mu2) )
    x0 = 1/2*(mu1+mu2) - np.log(p1/p2)*(mu1-mu2)/np.dot( mu1-mu2, np.dot(Si,mu1-mu2) )

    w = np.insert(w,0, -np.dot(w,x0) )

    return w





from collections import namedtuple
twolp = namedtuple("twolp", "W1 W2 f df")

def twolp_generate(training, labels, n,f,df, rho, **kwargs):
    x = np.hstack((np.ones((len(training),1)),training))
    y = np.array(labels)
    y[y==-1] = 0

    mom = kwargs.get('momentum', False)
    K = kwargs.get('iterations', 300)
    load = kwargs.get('load', False)

    input_dim = x.shape[1]
    f = np.vectorize(f)
    df = np.vectorize(df)

    # Input to layer is a row vector
    # Output is thus also a row vector
    # Dim of output of prior layer is col dim of matrix in the next

    if load:
        W1 = np.load(load+"_1.npy")
        W2 = np.load(load+"_2.npy")
    else:
        W1 = 0.001*np.random.rand(n,input_dim) # takes input vectors and returns output with n dim (neurons in first layer)
        W2 = 0.001*np.random.rand(1,n+1) # takes n and returns 1 (last layer)
    dW1  = 0
    dW2  = 0

    def run_through(data):
        v1 = np.dot(W1,data)
        y1 = f(v1)
        y1 = np.vstack((np.ones((1,len(y1[0]))),y1))

        v2 = np.dot(W2,y1)
        y2 = f(v2)
        return (v1,y1,v2,y2)

    cost = np.zeros( K+1 )
    #Training loop
    for k in range(0,K+1):
        ind = np.random.permutation(len(x))
        x = x[ind]
        y = y[ind]

        v1,y1,v2,y2 = run_through(x.T)


        d = np.square(y2-y.T)
        c = 1/2*np.asscalar(d.sum())
        cost[k] = c
        if k % 10 == 0:
            print(k)
            print("MEAN DIFF: " +str(d.mean()))
            print("COST: "+str(c))


        delta2 = df(v2) * (y2-y.T)
        dJ2 = np.dot(delta2, y1.T)
        dW2 = mom*dW2 - rho*dJ2 if mom else -rho*dJ2

        W2 = W2 + dW2


        delta1 = df(v1) * np.dot(delta2.T,np.delete(W2,0,1)).T
        dJ1 = np.dot(delta1, x)
        dW1 = mom*dW1 - rho*dJ1 if mom else -rho*dJ1
        W1 = W1 + dW1


    return W1,W2,cost




def twolp_classify(twolp,x):
    W1 = twolp.W1
    W2 = twolp.W2
    func = np.vectorize(twolp.f)

    v1 = np.dot(W1,x)
    y1 = augument(func(v1))
    v2 = np.asscalar(np.dot(W2,y1))
    y2 = func(v2)

    if y2 < 0.5: return -1
    elif y2 > 0.5: return 1
    else: return 0

def twolp_test(net,testing,labels):

    hit = 0
    miss = 0

    for i in range(0,len(testing)):
        x = augument(testing[i])
        y = labels[i]

        yh = twolp_classify(net,x)
        #print(yh)
        if y == yh: hit = hit + 1
        else: miss = miss + 1

    l = len(testing)
    summary = (
        hit/l,
        miss/l
    )

    return summary[0]







def linear_smo(X,Y):
    kernel = linear_kernel(X)
    alphas, bes = smo_simplified(kernel,Y)

    return alphas,bes

def smo_classify(a,b,X,Y,x):

    f = sum([ a[i]*Y[i]*np.dot(X[i],x) for i in range(len(a)) ]) + b

    if f < 0: return -1
    elif f > 0: return 1
    else: return 0

def smo_test(a,b,X,Y, testing,labels):
    hit = 0
    miss = 0

    for i in range(0,len(testing)):
        x = testing[i]
        y = labels[i]

        yh = smo_classify(a,b,X,Y,x)
        #print(yh)
        if y == yh: hit = hit + 1
        else: miss = miss + 1

    l = len(testing)
    summary = (
        hit/l,
        miss/l
    )

    return summary[0]
