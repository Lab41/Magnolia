from sklearn.decomposition import FastICA

def ica(X, sigdim=0, n_components=3):

    if sigdim==0:
        Xm = X
    else:
        Xm = X.T
            
    ica = FastICA(n_components=n_components)
    S_ = ica.fit_transform(Xm)
    A_ = ica.mixing_
    m_ = ica.mean_
    
    return A_, m_