''' spectral_features.py

Functions for featurizing audio using spectral analysis, as well as for
reconstructing time-domain audio signals from spectral features
'''

import scipy

def stft(x, fs, framesz, hop, two_sided=True):
    '''
    Short Time Fourier Transform (STFT) - Spectral decomposition

    Input:
        x - signal (1-d array, which is amp/sample)
        fs - sampling frequency (in Hz)
        framesz - frame size (in seconds)
        hop - skip length (in seconds)
        two_sided - not implemented

    Output:
        X = 2d array time-frequency repr of x, time x frequency
    '''
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp])
                     for i in range(0, len(x)-framesamp, hopsamp)])

    return X

def istft(X, fs, recon_size, hop):
    ''' Inverse Short Time Fourier Transform (iSTFT) - Spectral reconstruction

    Input:
        X - set of 1D time-windowed spectra, time x frequency
        fs - sampling frequency (in Hz)
        recon_size - total length of reconstruction willing to be performed
        hop - skip rate

    Output:
        x - a 1-D array holding reconstructed time-domain audio signal
    '''
    x = scipy.zeros(recon_size*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x
