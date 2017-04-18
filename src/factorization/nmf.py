import numpy as np



def nmf(X, k, maxiter=1000):
    '''
    Non-negative matrix factorization
    Input: X
    Output:
      - W = Components
      - H = Coefficients
     '''

    eps=1.0e-6

    # Convert to Python
    F = X.shape[0];
    T = X.shape[1];

    # Random initialization
    W = np.random.rand(F, k);
    H = np.random.rand(k, T);

    ONES = np.ones((F,T));

    for i in range(maxiter):
        # update activations
        H = H * (W.T  @ ( X / (W @ H+eps))) / (W.T @ ONES);

        # update dictionaries
        W = W * ((X / (W @ H+eps)) @ H.T) / (ONES @ H.T)

    return W, H

def snmf(X, k, sparsity=0.1, num_iters=100, W=None, H=None, W_init=None, H_init=None, W_norm='1', H_norm=None):
    '''
    Sparse Non-negative Matrix Factorization
    This function adapted from Graham Grindlay (grindlay@ee.columbia.edu)

    http://www.ee.columbia.edu/~grindlay/code.html

    Args:
        X (np.ndarray): 2-D n-by-m array to decompose
        k (int): number of components to use
        sparsity (float): sparsity constraint; higher values increase bias
            Common values include 0 (vanilla NMF), 0.01, 0.1
        num_iters (int): number of iterations to run parameter updates for
        W, H (np.ndarray): if not None, use W (n-by-k) and/or H (k-by-m) as given
            and do not perform any updates to it
        W_init, H_init (np.ndarray): if not None, use these as starting values
            for W and H
        W_norm (str): How to normalize W; '1' for L1-norm, '2' for L2-norm
        H_norm (str): if not None, how to normalize H

    Returns:
        W, H (np.ndarray) - dictionary and (sparse) mixing weights
    '''
    if np.min(X) < 0:
        raise ValueError('Non-negative values only')

    if np.any(np.sum(X,1) == 0):
        raise ValueError('Not all entries in a row can be zero')


    n, m = X.shape
    myeps = 10e-7

    # initialize W
    if W is None:
        if W_init is None:
            W = np.random.rand(n,k)
        else:
            W = W_init
        update_W = True
    else:
        update_W = False

    # initialize H
    if H is None:
        if H_init is None:
            H = np.random.rand(k,m)
        else:
            H = H_init
        update_H = True
    else:
        update_H = False

    # normalize W
    if W_norm=='1':
        def normalize_W(X):
            return np.apply_along_axis(
                lambda col: col/np.linalg.norm(col, ord=1), 1, X)
    elif W_norm=='2':
        def normalize_W(X):
            return np.apply_along_axis(
                lambda col: col/np.linalg.norm(col, ord=2), 1, X)
    W = normalize_W(W)

    # Normalize H
    if H_norm=='1':
        def normalize_H(X):
            return np.apply_along_axis(
                lambda row: row/np.linalg.norm(row, ord=1), 0, X)
    elif H_norm=='2':
        def normalize_H(X):
            return np.apply_along_axis(
                lambda row: row/np.linalg.norm(row, ord=2), 0, X)
    elif H_norm is None:
        def normalize_H(X):
            return X

    H = normalize_H(H)

    # preallocate matrix of ones
    Onn = np.ones((n,n))
    Onm = np.ones((n,m))

    # I_errs = []
    # s_errs = []
    # errs = []
    for t in range(num_iters):
        # update H if requested
        if update_H:
            k_by_m_penalty = W.T@Onm + sparsity
            H = H * ((W.T @ (X / (W@H))) /
                np.where(k_by_m_penalty>myeps, k_by_m_penalty, myeps))
            if H_norm:
                H = normalize_H(H)

        # update W if requested
        if update_W:
            R = X/(W@H)
            if W_norm == '1':
                n_by_k_term = Onm@H.T + (Onn@(R@H.T * W))
                n_by_k_term = np.where(n_by_k_term>myeps, n_by_k_term, myeps)
                W = W * ((R@H.T + (Onn@(Onm@H.T * W))) / n_by_k_term)
            elif W_norm == 2:
                n_by_k_term = Onm@H.T + W * (Onn@(R@H.T * W))
                W = W * ((R@H.T + W * (Onn@(Onm@H.T * W))) /
                           np.where(n_by_k_term>myeps, n_by_k_term, myeps))
            W = normalize_W(W)

        # TODO: add error calculation and convergence checks
    return W, H
