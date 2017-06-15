import numpy as np
from sklearn.cluster import KMeans



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

def snmf(X, k, sparsity=0.1, num_iters=100, W=None, H=None, W_init=None, H_init=None, W_norm='1', H_norm=None,
    return_errors=False, update_weight=1.0):
    '''
    Sparse Non-negative Matrix Factorization
    This function adapted from Graham Grindlay (grindlay@ee.columbia.edu)
    (nmf_kl_sparse_es)

    http://www.ee.columbia.edu/~grindlay/code.html

    To train dictionaries, send in data from individual speakers on X and
    collect the W matrices returned.
    To do inference on mixtures, concatenate the appropriate speaker dictionaries
    and send them in on W, with mixture data on X. The separated components
    will be in the products W[:, i:j] @ H[i:j, :], where i and j index the components
    in the concatenated dictionary corresponding to a particular speaker.

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
    myeps = 10e-8

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

    I_errs = []
    s_errs = []
    errs = []
    for t in range(num_iters):
        # update H if requested
        if update_H:
            WH = W@H + myeps
            k_by_m_penalty = W.T@Onm + sparsity
            H = ((1. - update_weight) * H +
                update_weight * (H * ((W.T @ (X / WH)) /
                np.where(k_by_m_penalty>myeps, k_by_m_penalty, myeps))))
            if H_norm:
                H = normalize_H(H)

        # update W if requested
        if update_W:
            WH = W@H + myeps
            R = X/WH
            if W_norm == '1':
                n_by_k_term = Onm@H.T + (Onn@(R@H.T * W))
                n_by_k_term = np.where(n_by_k_term>myeps, n_by_k_term, myeps)
                W = ((1. - update_weight) * W +
                    update_weight * (W * ((R@H.T + (Onn@(Onm@H.T * W))) / n_by_k_term)))
            elif W_norm == 2:
                n_by_k_term = Onm@H.T + W * (Onn@(R@H.T * W))
                W = ((1. - update_weight) * W +
                    update_weight * (W * ((R@H.T + W * (Onn@(Onm@H.T * W))) /
                           np.where(n_by_k_term>myeps, n_by_k_term, myeps))))
            W = normalize_W(W)

        # TODO: add error calculation and convergence checks
        #
        # % compute squared error
        # I_errs(t) = sum(V(:).*log(V(:)./R(:)) - V(:) + R(:));
        # s_errs(t) = sum(H(:));
        # errs(t) = I_errs(t) + alpha*s_errs(t);
        WH = (W@H) + myeps
        # I_errs.append(np.sum(X * np.log(X / WH + myeps) - X + WH))
        I_errs.append(np.sum(np.square(X-WH)))
        s_errs.append(np.sum(H))
        errs.append(I_errs[-1] + sparsity * s_errs[-1])

    if return_errors:
        return W, H, I_errs, s_errs, errs
    else:
        return W, H

def nmf_separate(mix, spkr_models, mask=False, num_iters=500):
    '''
    Args:
        mix (array-like): matrix (magnitude spectrogram in F x T) to separate.
        spkr_models (list): paired W,H matrices for each speaker. H can be any value; it is not used
        mask (bool): return matrix products directly or use them as ratio masks on mix?
    '''
    w_all = np.concatenate([x[0] for x in spkr_models], axis=1)
    new_h_all = []
    reconstructions = []
    num_spkrs = len(spkr_models)
    total_components = w_all.shape[1]
    components_per_spkr = total_components // num_spkrs
    _, h_new = snmf(mix, total_components, num_iters=num_iters, sparsity=0.0, W=w_all)
    for i in range(num_spkrs):
        h_new_spkr = h_new[i*components_per_spkr:(i+1)*components_per_spkr]
        new_h_all.append(h_new_spkr)
        reconstruction = spkr_models[i][0] @ h_new_spkr
        reconstructions.append(reconstruction)
    if mask:
        masks = np.array(reconstructions)
        norm_mask = masks.sum(axis=0)
        for i in range(num_spkrs):
            masks[i] = masks[i] / norm_mask
            reconstructions[i] = masks[i] * mix
    return reconstructions

def easy_nmf_separate(X, num_spkrs = 2, k = 20, freq_comps = (60,100), sparsity=0.001):
    '''
    All-in-one separation code based on NMF. Uses K-means clustering on
    frequency-domain basis vectors 

    Args:
        X: TF representation of input mixture (time x frequency)
        num_spkrs: number of components to return
        k: Number of NMF bases to estimate
        freq_comps: tuple, low and high bounds of which frequency components to
            cluster on
        sparsity: see snmf

    Returns:
        array-like with separated TF signals on the zeroth axis
    '''

    W, H = snmf(np.sqrt(np.abs(X)).T, k, sparsity=sparsity)

    # K-means on weights
    # cluster on certain frequency components
    a, b = freq_comps
    km = KMeans(num_spkrs)
    km.fit(W[a:b,:].T)
    cluster_memberships = [np.nonzero(km.labels_ == i)[0] for i in range(num_spkrs)]

    recons = [(W[:, cl_idxs] @ H[cl_idxs, :]).T for cl_idxs in cluster_memberships]

    return np.stack(recons)
