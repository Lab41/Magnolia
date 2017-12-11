import numpy as np

def pca(X,sigdim=1):
    '''
    pca( X, k= None, sigdim=0/1 )
    
    sigdim = the dimension along which the signal is aligned:     
            
            For example, if I have 5 signals of length 1000, then
                - for sigdim = 0, X is 1000 x 5
                - for sigdim = 1, X is 5 x 1000
    
    '''    
    if sigdim==0:
        m = X.mean(axis=0)
        Xm = X - m
    else:
        m = X.mean(axis=1)
        Xm = (X.T - m)
        
    C = Xm.T.dot(Xm)/len(Xm)
    
    return np.linalg.eig(C) + (m,)