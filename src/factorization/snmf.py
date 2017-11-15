import numpy as np
import numpy.matlib
import h5py


def extend_features(V, T_L=0, T_R=0):
    M = V.shape[0]
    N = V.shape[1]

    V_extended = np.empty((M*(T_L + T_R + 1), N))
    for i in range(T_L):
        V_extended[i*M:(i+1)*M, :] = np.pad(V, ((0,0),(T_L - i,0)), mode='edge')[:, :-(T_L - i)]
    V_extended[T_L*M:(T_L + 1)*M, :] = V
    for i in range(T_R):
        V_extended[(T_L + 1 + i)*M:(T_L + 2 + i)*M, :] = np.pad(V,((0,0),(0,(i + 1))), mode='edge')[:, (i + 1):]

    return V_extended


def sparse_nmf(V, R=None,
               max_iter=100, conv_eps=1e-3, rng=None,
               sparsity=0, cf=None, beta=1.0,
               init_W=None, init_H=None,
               W_update_ind=None, H_update_ind=None,
               verbose=False):
    """Sparse NMF with beta-divergence reconstruction error,
     L1 sparsity constraint, optimization in normalized basis vector space.

     Inputs:
     V:  M x N matrix to be factorized
     params: optional parameters
         beta:     beta-divergence parameter (default: 1, i.e., KL-divergence)
         cf:       cost function type (default: 'kl'; overrides beta setting)
                   'is': Itakura-Saito divergence
                   'kl': Kullback-Leibler divergence
                   'ed': Euclidean distance
         sparsity: weight for the L1 sparsity penalty (scalar, vector (R, 1), matrix (R, N)) (default: 0)
         max_iter: maximum number of iterations (default: 100)
         conv_eps: threshold for early stopping (default: 0,
                                                 i.e., no early stopping)
         verbose:  display evolution of objective function (default: False)
         rng: set the random seed to the given value
         init_W:   initial setting for W (default: random;
                                          either init_w or r have to be set)
         R:        number of basis functions (default: based on init_w's size;
                                              either init_w or r have to be set)
         init_H:   initial setting for H (default: random)
         W_update_ind: set of dimensions to be updated (default: all)
         H_update_ind: set of dimensions to be updated (default: all)

     Outputs:
     W: matrix of basis functions
     H: matrix of activations
     objective: objective function values throughout the iterations



     References:
     J. Eggert and E. Korner, "Sparse coding and NMF," 2004
     P. D. O'Grady and B. A. Pearlmutter, "Discovering Speech Phones
       Using Convolutive Non-negative Matrix Factorisation
       with a Sparseness Constraint," 2008
     J. Le Roux, J. R. Hershey, F. Weninger, "Sparse NMF - half-baked or well
       done?," 2015

     This implementation follows the derivations in:
     J. Le Roux, J. R. Hershey, F. Weninger,
     "Sparse NMF - half-baked or well done?,"
     MERL Technical Report, TR2015-023, March 2015

     If you use this code, please cite:
     J. Le Roux, J. R. Hershey, F. Weninger,
     "Sparse NMF - half-baked or well done?,"
     MERL Technical Report, TR2015-023, March 2015
       @TechRep{LeRoux2015mar,
         author = {{Le Roux}, J. and Hershey, J. R. and Weninger, F.},
         title = {Sparse {NMF} - half-baked or well done?},
         institution = {Mitsubishi Electric Research Labs (MERL)},
         number = {TR2015-023},
         address = {Cambridge, MA, USA},
         month = mar,
         year = 2015
       }

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       Copyright (C) 2015 Mitsubishi Electric Research Labs (Jonathan Le Roux,
                                             Felix Weninger, John R. Hershey)
       Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    M = V.shape[0]
    N = V.shape[1]

    if rng is None:
        rng = np.random

    if cf == 'kl':
        beta = 1
    elif cf == 'is':
        beta = 0
    elif cf == 'ed':
        beta = 2

    W = None
    if init_W is None:
        if R is None:
            print('FATAL ERROR')
            exit()
        W = rng.rand(M, R)
    else:
        Ri = init_W.shape[1]
        W = np.empty((M, R))
        W[:, :Ri] = init_W
        if Ri < R:
            W[:, Ri:R] = rng.rand(M, R - Ri)
        else:
            R = Ri

    H = None
    if init_H is None:
        H = rng.rand(R, N)
    elif isinstance(init_H, str) and init_H == 'ones':
        H = np.ones((R, N), dtype=float)
    else:
        H = init_H

    if W_update_ind is None:
        W_update_ind = np.ones((R), dtype=np.bool_)
    if H_update_ind is None:
        H_update_ind = np.ones((R), dtype=np.bool_)

    # sparsity per matrix entry
    if np.isscalar(sparsity) or len(sparsity) == 1:
        sparsity = sparsity*np.ones((R, N), dtype=float)
    elif sparsity.shape[1] == 1:
        sparsity = np.matlib.repmat(sparsity, 1, N)

    # normalize the columns of W and rescale H accordingly
    Wn = np.sqrt(np.sum(W**2, axis=0))
    W /= Wn
    H *= Wn[:, None]

    eps = np.finfo(float).eps
    flr = 1e-9
    lambda_ = np.dot(W, H)
    np.maximum(lambda_, flr, out=lambda_)
    last_cost = np.inf

    objective = {}
    objective['div'] = np.zeros(max_iter)
    objective['cost'] = np.zeros(max_iter)

    div_beta = beta
    H_ind = H_update_ind
    W_ind = W_update_ind
    update_H = np.sum(H_ind)
    update_W = np.sum(W_ind)

    if verbose:
        print('Performing sparse NMF with beta-divergence, beta = {}'.format(div_beta))

    H_factor = None
    W_factor = None
    MN_array = np.empty_like(V)
    for it in range(max_iter):
        # H updates
        if update_H > 0:
            if div_beta == 1:
                # dmh = np.dot(W[:, H_ind].T, V/lambda_)
                # dph = np.sum(W[:, H_ind], axis=0)[:, None] + sparsity[H_ind, :]
                # dph = np.maximum(dph, flr)
                R_length_array = np.empty(update_H, dtype=W.dtype)
                np.divide(V, lambda_, out=MN_array)
                H_factor = np.dot(W[:, H_ind].T, MN_array)
                np.sum(W[:, H_ind], axis=0, out=R_length_array)
                H_factor /= np.maximum(R_length_array[:, None] + sparsity[H_ind, :], flr)
            elif div_beta == 2:
                # dmh = np.dot(W[:, H_ind].T, V)
                # dph = np.dot(W[:, H_ind].T, lambda_) + sparsity[H_ind, :]
                # dph = np.maximum(dph, flr)
                RN_array = np.empty((update_H, V.shape[1]), dtype=V.dtype)
                np.dot(W[:, H_ind].T, V, out=RN_array)
                H_factor = RN_array.copy()
                np.dot(W[:, H_ind].T, lambda_, out=RN_array)
                H_factor /= np.maximum(RN_array + sparsity[H_ind, :], flr)
            else:
                # dmh = np.dot(W[:, H_ind].T, V*(lambda_**(div_beta - 2)))
                # dph = np.dot(W[:, H_ind].T, lambda_**(div_beta - 1)) + sparsity[H_ind, :]
                # dph = np.maximum(dph, flr)
                RN_array = np.empty((update_H, V.shape[1]), dtype=V.dtype)
                np.power(lambda_, (div_beta - 2), out=MN_array)
                MN_array *= V
                np.dot(W[:, H_ind].T, MN_array, out=RN_array)
                H_factor = RN_array.copy()
                np.power(lambda_, (div_beta - 1), out=MN_array)
                np.dot(W[:, H_ind].T, MN_array, out=RN_array)
                H_factor /= np.maximum(RN_array + sparsity[H_ind, :], flr)

            H[H_ind, :] *= H_factor
            np.dot(W, H, out=lambda_)
            np.maximum(lambda_, flr, out=lambda_)

        # W updates
        if update_W > 0:
            MR_array = np.empty((V.shape[0], update_W), dtype=V.dtype)
            MR_array2 = np.empty((V.shape[0], update_W), dtype=V.dtype)
            R_length_array = np.empty(update_W, dtype=W.dtype)
            R_length_array2 = np.empty(update_W, dtype=W.dtype)
            if div_beta == 1:
                # dmw = np.dot(V/lambda_, H[W_ind, :].T) + np.sum(np.sum(H[W_ind, :], axis=1).T*W[:, W_ind], axis=0)*W[:, W_ind]
                # dpw = np.sum(H[W_ind, :], axis=1) + np.sum(np.dot(V/lambda_, H[W_ind, :].T)*W[:, W_ind], axis=0)*W[:, W_ind]
                # dpw = np.maximum(dpw, flr)
                np.divide(V, lambda_, out=MN_array)
                np.dot(MN_array, H[W_ind, :].T, out=MR_array)
                W_factor = MR_array.copy()
                np.sum(H[W_ind, :], axis=1, out=R_length_array)
                np.multiply(R_length_array.T, W[:, W_ind], out=MR_array)
                np.sum(MR_array, axis=0, out=R_length_array)
                np.multiply(R_length_array, W[:, W_ind], out=MR_array)
                W_factor += MR_array
                np.sum(H[W_ind, :], axis=1, out=R_length_array)
                np.divide(V, lambda_, out=MN_array)
                np.dot(MN_array, H[W_ind, :].T, out=MR_array)
                np.multiply(MR_array, W[:, W_ind], out=MR_array)
                np.sum(MR_array, axis=0, out=R_length_array2)
                np.multiply(R_length_array2, W[:, W_ind], out=MR_array)
                W_factor /= np.maximum(R_length_array + MR_array, flr)
            elif div_beta == 2:
                dmw = np.dot(V, H[W_ind, :].T) + np.sum(np.dot(lambda_, H[W_ind, :].T)*W[:, W_ind], axis=0)*W[:, W_ind]
                dpw = np.dot(lambda_, H[W_ind, :].T) + np.sum(np.dot(V, H[W_ind, :].T)*W[:, W_ind], axis=0)*W[:, W_ind]
                dpw = np.maximum(dpw, flr)
            else:
                dmw = np.dot(V*lambda_**(div_beta - 2), H[W_ind, :].T) + np.sum(np.dot(lambda_**(div_beta - 1), H[W_ind, :].T)*W[:, W_ind], axis=0)*W[:, W_ind]
                dpw = np.dot(lambda_**(div_beta - 1), H[W_ind, :].T) + np.sum(np.dot(V*lambda_**(div_beta - 2), H[W_ind, :].T)*W[:, W_ind], axis=0)*W[:, W_ind]
                dpw = np.maximum(dpw, flr)
            # W[:, W_ind] *= dmw
            # W[:, W_ind] /= dpw
            W[:, W_ind] *= W_factor

            # normalize the columns of W
            W /= np.sqrt(np.sum(W**2, axis=0))
            np.dot(W, H, out=lambda_)
            np.maximum(lambda_, flr, out=lambda_)

        # compute the objective function
        if div_beta == 1:
            # div = np.sum(V*np.log(V/lambda_) - V + lambda_)
            # NOTE: for numerical stability
            lambda_[lambda_ <= 0.0] = eps
            np.divide(V, lambda_, out=MN_array)
            # NOTE: for numerical stability
            MN_array[MN_array <= 0.0] = eps
            np.log(MN_array, out=MN_array)
            np.multiply(V, MN_array, out=MN_array)
            np.subtract(MN_array, V, out=MN_array)
            np.add(MN_array, lambda_, out=MN_array)
            # NOTE: for numerical stability
            # div = np.sum(MN_array)
            div = np.sum(MN_array/MN_array.shape[0], axis=0)
            div = np.sum(div/MN_array.shape[1])
        elif div_beta == 2:
            div = np.sum((V - lambda_)**2)
        elif div_beta == 0:
            div = np.sum(V/lambda_ - np.log(V/lambda_) - 1.0)
        else:
            div = np.sum(V**div_beta + (div_beta - 1.0)*lambda_**div_beta - div_beta*V*lambda_**(div_beta - 1.0))/(div_beta*(div_beta - 1.0))
        # NOTE: for numerical stability
        # cost = div + np.sum(sparsity*H)
        cost = np.sum(sparsity*H/lambda_.shape[0], axis=0)
        cost = np.sum(cost/lambda_.shape[1])
        cost += div

        objective['div'][it] = div
        objective['cost'][it] = cost

        if verbose:
            print('iteration {} div = {:6.3f} cost = {:6.3f}'.format(it + 1, div, cost))

        # convergence check
        if it > 1 and conv_eps > 0.0:
            e = np.abs(cost - last_cost)/last_cost
            if (e < conv_eps):
                if verbose:
                    print('Convergence reached, aborting at iteration {}'.format(it + 1))
                objective['div'] = objective['div'][:it]
                objective['cost'] = objective['cost'][:it]
                break

        last_cost = cost

    return H, W, objective


class SNMF:
    def __init__(self, T_L=0, T_R=0, R=None,
                 sparsity=0, cf=None, beta=1.0):
        self._T_L = T_L
        self._T_R = T_R
        self._R = R
        self._sparsity = sparsity
        self._cf = cf
        self._beta = beta
        self._source_library = {}

    def load(self, checkpoint_filename):
        library_file = h5py.File(checkpoint_filename, 'r')

        for source in library_file.keys():
            self._source_library[source] = library_file[source][:, :]

        library_file.close()

    def save(self, library_output_file):
        library_file = h5py.File(library_output_file, 'w')

        for source in self._source_library:
            ds = library_file.create_dataset(str(source),
                                             data=self._source_library[source])

        library_file.close()

    def update(self, spec, source,
               max_iter=100, conv_eps=1e-3, rng=None,
               init_W=None, init_H=None,
               W_update_ind=None, H_update_ind=None,
               verbose=False):

        if init_W is None and source in self._source_library:
            init_W = self._source_library[source]

        V = extend_features(np.abs(spec), T_L=self._T_L, T_R=self._T_R)

        H, W, obj = sparse_nmf(V, R=self._R,
                               max_iter=max_iter, conv_eps=conv_eps, rng=rng,
                               sparsity=self._sparsity, cf=self._cf, beta=self._beta,
                               init_W=init_W, init_H=init_H,
                               W_update_ind=W_update_ind, H_update_ind=H_update_ind,
                               verbose=verbose)

        self._source_library[source] = W

        return H, W, obj
    
    def batch_update(self, spec_batch, source_label,
                     max_iter=100, conv_eps=1e-3, rng=None,
                     init_W=None, init_H=None,
                     W_update_ind=None, H_update_ind=None,
                     verbose=False):

        if init_W is None and source_label in self._source_library:
            init_W = self._source_library[source_label]
        
        for i, spec in enumerate(spec_batch):
            spec_batch[i] = extend_features(np.abs(spec), T_L=self._T_L, T_R=self._T_R)
        V = np.hstack(spec_batch)

        H, W, obj = sparse_nmf(V, R=self._R,
                               max_iter=max_iter, conv_eps=conv_eps, rng=rng,
                               sparsity=self._sparsity, cf=self._cf, beta=self._beta,
                               init_W=init_W, init_H=init_H,
                               W_update_ind=W_update_ind, H_update_ind=H_update_ind,
                               verbose=verbose)

        self._source_library[source_label] = W

        return H, W, obj

    def source_separate(self, spec,
                        max_iter=100, conv_eps=1e-3, rng=None,
                        init_H=None, H_update_ind=None,
                        verbose=False):
        result = {}
        V_sum = None

        M_original = spec.shape[0]
        V = extend_features(np.abs(spec), T_L=self._T_L, T_R=self._T_R)
        MN_array = np.empty_like(V)

        for source in self._source_library:
            init_W = self._source_library[source]

            H, W, obj = sparse_nmf(V, R=self._R,
                                   max_iter=max_iter, conv_eps=conv_eps, rng=rng,
                                   sparsity=self._sparsity, cf=self._cf, beta=self._beta,
                                   init_W=init_W, init_H=init_H,
                                   W_update_ind=np.zeros(self._R, dtype=np.bool_),
                                   H_update_ind=H_update_ind,
                                   verbose=verbose)

            np.dot(init_W, H, out=MN_array)
            result[source] = MN_array.copy()

            if V_sum is None:
                V_sum = MN_array.copy()
            else:
                V_sum += MN_array

        np.divide(V, V_sum, out=MN_array)
        for source in self._source_library:
            result[source] *= MN_array
            result[source] = result[source][self._T_L*M_original:(self._T_L + 1)*M_original, :]
            result[source] = result[source]*np.exp(1j*np.angle(spec))

        return result
    
    @staticmethod
    def temporal_mask_for_best_features(source_spec):
        max_along_time_bins = np.log(np.amax(np.abs(source_spec[6:, :]), axis=0))
        m = max_along_time_bins - max_along_time_bins.mean() > -.5*max_along_time_bins.std()
        return m
    
    @staticmethod
    def split_along_time_bins(spec, m):
        m_shift = np.roll(m, 1)
        m_shift[0] = False
        indices=np.arange(m.size)[np.logical_xor(m, m_shift)]
        if indices[0] == 0:
            indices = indices[1:]
        all_splits=np.split(spec, indices, axis=1)
        splits=[]
        keep_split=m[0]
        for i in range(len(all_splits)):
            if keep_split:
                splits.append(all_splits[i])
            keep_split = not keep_split
        return splits
