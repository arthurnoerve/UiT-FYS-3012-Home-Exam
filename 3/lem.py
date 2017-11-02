






# Laplacian eigen map implementation





# Construct Adjacency matrix


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
            ds = [np.norm(x-z)**2 for z in X]
            i = np.argsort(ds)
            max_d = ds[i][n]
            return l < max_d


    if t:
        def get_weight(l):
            return np.exp(-l/t)
    else:
        def get_weight(l):
            return 1

    dim = len(x)
    A = np.zeros((dim,dim))

    for i in range(len(x)):
        x = X[i]
        for j in range(len(x)):
            y = X[j]
            if x == y: continue
            l = np.norm(x-y)**2
            if is_connected(X,l,x,y):
                A[i,j] = get_weight(l)

    # Solve problem
    D = sum(A, axis=0)
    L = D - A

    l,v = scipy.linalg.eig(L,D)

    idx = l.argsort()
    l = l[idx]
    v = v[:,idx]

    return v
