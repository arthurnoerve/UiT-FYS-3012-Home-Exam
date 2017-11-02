import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
from scipy.spatial.distance import pdist, squareform

def linear_kernel(Xtr, Xte = None, kpar = None):
    """
    Calculate linear kernel.
    Args:
        Xtr: Numpy matrix of training data with shape (Ntr, d)
        Xte: Numpy matrix of test data with shape (Nte, d). (OPTIONAL)
        kpar: Not used.

    Returns:
        Ktr: Gaussian kernel within training data. Shape: (Ntr, Ntr)
        Kte: Gaussian kernel between test and training data. shape: (Nte, Ntr)
             NOTE: This is only returned if test data is provided to the function.
    """
    Xtr = np.asarray(Xtr)
    Ktr = Xtr.dot(Xtr.T)

    if Xte is None:
        return Ktr
    
    Xte = np.asarray(Xte)
    Kte = Xte.dot(Xtr.T)

    return Ktr, Kte

def gaussian_kernel(Xtr, Xte = None, kpar = 0.1):
    """
    Calculate Gaussian kernel by.
        k(x1, x2) = exp(-0.5/kpar**2*||x1-x2||**2)

    Args:
        Xtr: Numpy matrix of training data with shape (Ntr, d)
        Xte: Numpy matrix of test data with shape (Nte, d). (OPTIONAL)
        kpar: Width parameter.

    Returns:
        Ktr: Gaussian kernel within training data. Shape: (Ntr, Ntr)
        Kte: Gaussian kernel between test and training data. shape: (Nte, Ntr)
             NOTE: This is only returned if test data is provided to the function.
    """
    distmat_tr = seucdist(Xtr, Xtr)
    Ktr = np.exp(-0.5/(kpar**2)*distmat_tr)

    if Xte is None:
        return Ktr

    distmat_te = seucdist(Xte, Xtr)
    Kte = np.exp(-0.5/(kpar**2)*distmat_te)

    return Ktr, Kte

def seucdist(X1, X2):
    """
    Calculate the squared Euclidean from the data points in X1 to the data points in X2.

    Args:
        X1: Numpy matrix containing data points with shape (N1, d)
        X2: Numpy matrix containing data points with shape (N2, d)

    Returns:
        distmat: Squared Euclidean distances. Shape: (N1, N2)
    """
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    n1, d = X1.shape
    n2, _ = X2.shape

    A = np.ones((d, n2))
    B = np.ones((n1, d))

    distmat = (X1**2).dot(A) - 2*X1.dot(X2.T) + B.dot((X2**2).T)

    return distmat

def plot_svm_train(Xtr, y, alpha, b, kernel_function = linear_kernel, kpar = None):
    """Plot SVM training data with margins and decision boundary.

    NOTE: 2D support only.

    Args:
        Xtr: Numpy matrix containing training data with shape (N, 2).
        y: Numpy vector of training labels with values in {-1, 1}. The shape of the vector should be (N,)
        alpha: Numpy vector of Lagrange multipliers with shape (N,)
        b: Bias
        kernel_function: Pointer to function that calculates kernel. The function should have the arguments (Xtr, Xte, kpar)
                         and return a touple (Ktr, Kte) containing kernel matrices.
        kpar: Parameter for kernel function.
    """
    if Xtr.shape[1] != 2:
        raise ValueError('Unsupported data dimensionality')

    # Generate a grid of datapoints.
    x1min = np.min(Xtr[:,0])
    x1max = np.max(Xtr[:,0])
    x1margin = 0.05*(x1max-x1min)

    x2min = np.min(Xtr[:,1])
    x2max = np.max(Xtr[:,1])
    x2margin = 0.05*(x2max-x2min)

    x1axis = np.linspace(x1min - x1margin, x1max + x1margin, 200)
    x2axis = np.linspace(x2min - x2margin, x2max + x2margin, 200)

    X1grid = np.tile(x1axis, (len(x2axis),1)).T.reshape(-1, 1)
    X2grid = np.tile(x2axis, (1, len(x1axis))).reshape(-1, 1)

    Xgrid = np.concatenate((X1grid, X2grid), axis=1)

    # Calculate kernel
    _, Kte = kernel_function(Xtr = Xtr, Xte = Xgrid, kpar = kpar)

    # Evaluate SVM classification
    f_x = np.atleast_2d(Kte.dot(alpha*y) + b)

    # Plot contour
    X1grid = X1grid.reshape(len(x2axis), -1)
    X2grid = X2grid.reshape(-1, len(x1axis))
    f_x = f_x.reshape(len(x2axis), len(x1axis))

    # Plot decision boundary and margins
    plt.contour(X1grid, X2grid, f_x, levels=[-1, 0, 1], linestyles=('dashed', 'solid', 'dashed'), linewidths=2, colors='k')

    # Contour. Ensure that min + max = 0 (for colormap)
    f_x[f_x < 0] = f_x[f_x < 0]/np.abs(np.min(f_x[f_x < 0]))
    f_x[f_x > 0] = f_x[f_x > 0]/np.max(f_x[f_x > 0])

    plt.contourf(X1grid, X2grid, f_x, levels=np.linspace(np.min(f_x), np.max(f_x),200), cmap='seismic')

    # Plot training data.
    mask_sv_c1 = (alpha > 0) & (y == 1)
    mask_nsv_c1 = (alpha == 0) & (y == 1)
    mask_sv_c2 = (alpha > 0) & (y == -1)
    mask_nsv_c2 = (alpha == 0) & (y == -1)

    # Support vectors
    plt.scatter(Xtr[mask_sv_c1,0], Xtr[mask_sv_c1,1], c='r', s=30, edgecolors='w')
    plt.scatter(Xtr[mask_sv_c2,0], Xtr[mask_sv_c2,1], c='b', s=30, edgecolors='w')

    # Other
    plt.scatter(Xtr[mask_nsv_c1,0], Xtr[mask_nsv_c1,1], c='r', s=30)
    plt.scatter(Xtr[mask_nsv_c2,0], Xtr[mask_nsv_c2,1], c='b', s=30)

    # Set axes
    plt.axis([x1axis[0], x1axis[-1], x2axis[0], x2axis[-1]])

def smo_simplified(K, y, C = 10, tol = 1e-6, conv_iter = 1000, max_iter = 10000):
    """Simplified SMO implementation.

    Optimizes the SVM optimization problem using a simplified
    Sequential Minimal Optimization algorithm. The implementation
    is based on [1]. 
    
    Note that the hyperplane is on the form f(x) = w'x + b.

    Args:
        K: Numpy matrix of pairwise inner products in the training set or kernel matrix.
           The shape of the matrix should be (N, N), where N is the number of data
           points in the training set.
        y: Numpy vector of labels with values in {-1, 1}. The shape of the vector
           should be (N,)
        C: Weight for the slack variables. A large C forces the margins to be more narrow.
        tol: Tolerance for updating Lagrange multipliers.
        conv_iter: Number of iterations without parameter updates before determining that the
                   algorithm has converged.
        max_iter: The maximum number of iterations before the algorithm terminates.

    Returns:
        alpha: Lagrange multipliers.
        b: Bias


    [1] http://cs229.stanford.edu/materials/smo.pdf

    """
    # Ensure that the inputs are numpy arrays
    y = np.asarray(y).flatten()
    K = np.asarray(K)

    N = len(y)

    if not K.shape == (N, N):
        raise ValueError('K should be an %d times %d matrix.' % (N, N))

    # Initialization
    alpha = np.zeros((N,), dtype=float)
    b = 0

    n_iter_conv = 0
    n_iter = 0

    while n_iter_conv < conv_iter and n_iter < max_iter:
        n_pairs_changed = 0

        for i in range(N):
            # Update if values are outside of the given tolerance
            E_i = calculate_E(i, K, alpha, y, b)
            if ((y[i]*E_i < -tol and alpha[i] < C) or (y[i]*E_i > tol and alpha[i] > 0)):
                # Select random j different from i
                j = i
                while j == i:
                    j = random.randint(0, N - 1)

                # Calculate lower and upper bounds
                L, H = calculate_L_H(i, j, y, alpha, C)

                if L == H:
                    continue

                eta = calculate_eta(i, j, K)

                if eta >= 0:
                    continue

                # Evaluate
                E_j = calculate_E(j, K, alpha, y, b)

                alpha_i_old = alpha[i].copy()
                alpha_j_old = alpha[j].copy()

                # Update alpha_j
                alpha[j] = update_alpha(alpha[j], y[j], E_i, E_j, eta, H, L)
                
                # No need to update alpha_i
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue

                # Update alpha_i and calculate new bias
                alpha[i] += y[i]*y[j]*(alpha_j_old - alpha[j])

                # Update bias
                b = calculate_b(b, i, j, K, C, y, alpha, E_i, E_j, alpha_i_old, alpha_j_old)

                n_pairs_changed += 1
        
        # Reset if some alpha pairs have changed
        if n_pairs_changed == 0:
            n_iter_conv += 1
        else:
            n_iter_conv = 0

        n_iter += 1

    if n_iter == max_iter:
        warnings.warn('The SMO algorithm did not converge after %d iterations.' % n_iter, RuntimeWarning)

    return alpha, b

def calculate_E(idx, K, alpha, y, b):
    return ((alpha*y).dot(K[:, idx]) + b - y[idx])

def calculate_L_H(idx_i, idx_j, y, alpha, C):
    y_i = y[idx_i]
    y_j = y[idx_j]
    alpha_i = alpha[idx_i]
    alpha_j = alpha[idx_j]

    if not (y_i == y_j):
        L = max((0, alpha_j - alpha_i))
        H = min((C, C + alpha_j - alpha_i))
    else:
        L = max((0, alpha_i + alpha_j - C))
        H = min((C, alpha_i + alpha_j))

    return (L, H)

def calculate_eta(idx_i, idx_j, K):
    return (2*K[idx_i, idx_j] - K[idx_i, idx_i] - K[idx_j, idx_j])

def update_alpha(alpha_old, y, E_i, E_j, eta, H, L):
    
    # Update alpha
    alpha = alpha_old - y*(E_i - E_j)/eta
    
    # Clamp to valid values
    if alpha > H:
        alpha = H
    elif alpha < L:
        alpha = L
    
    return alpha

def calculate_b(b, idx_i, idx_j, K, C, y, alpha, E_i, E_j, alpha_i_old, alpha_j_old):
    b1 = b - E_i - y[idx_i]*(alpha[idx_i] - alpha_i_old)*K[idx_i, idx_i] - y[idx_j]*(alpha[idx_j] - alpha_j_old)*K[idx_i, idx_j]
    b2 = b - E_j - y[idx_i]*(alpha[idx_i] - alpha_i_old)*K[idx_i, idx_j] - y[idx_j]*(alpha[idx_j] - alpha_j_old)*K[idx_j, idx_j]

    # Ensure that KKT are not violated
    if 0 < alpha[idx_i] < C:
        b = b1
    elif 0 < alpha[idx_j] < C:
        b = b2
    else:
        b = (b1 + b2)/2

    return b