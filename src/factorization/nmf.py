import numpy as np

'''
Non-negative matrix factorization
Input: X
Output:
  - W = Components
  - H = Coefficients
 '''

def nmf(X, k, maxiter=1000):

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
        W = W * ((X /(W @ H+eps)) @ H.T) / (ONES @ H.T)

    return W, H
