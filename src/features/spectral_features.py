''' spectral_features.py

Functions for featurizing audio using spectral analysis, as well as for
reconstructing time-domain audio signals from spectral features
'''

import numpy as np
import scipy

def stft(x, fs, framesz, hop, two_sided=True, fft_size=None):
    '''
    Short Time Fourier Transform (STFT) - Spectral decomposition

    Input:
        x - signal (1-d array, which is amp/sample)
        fs - sampling frequency (in Hz)
        framesz - frame size (in seconds)
        hop - skip length (in seconds)
        two_sided - return full spectrogram if True
            or just positive frequencies if False
        fft_size - length (in seconds) of DFT window

    Output:
        X = 2d array time-frequency repr of x, time x frequency
    '''

    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    # set size of FFT window
    if fft_size is None:
        fft_size = framesamp
    else:
        fft_size = int(fft_size * fs)
    w = scipy.hanning(framesamp)
    if two_sided:
        X = scipy.array([scipy.fft(w*x[i:i+framesamp], n=fft_size)
                     for i in range(0, len(x)-framesamp, hopsamp)])
    else:
        X = scipy.array([np.fft.fftpack.rfft(w*x[i:i+framesamp], n=fft_size)
             for i in range(0, len(x)-framesamp, hopsamp)])

    return X

def istft(X, fs, recon_size, hop, two_sided=True):
    ''' Inverse Short Time Fourier Transform (iSTFT) - Spectral reconstruction

    Input:
        X - set of 1D time-windowed spectra, time x frequency
        fs - sampling frequency (in Hz)
        recon_size - total length of reconstruction
        hop - skip rate between successive windows

    Output:
        x - a 1-D array holding reconstructed time-domain audio signal
    '''
    x = scipy.zeros(int(recon_size*fs))
    hopsamp = int(hop*fs)
    # TODO: do we need to mess with the framewise reconstruction size?
    if two_sided:
        framesamp = X.shape[1]
        inverse_transform = scipy.ifft
    else:
        framesamp = (X.shape[1] - 1) * 2
        inverse_transform = np.fft.fftpack.irfft

    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(inverse_transform(X[n]))
    return x

def reconstruct(spec_mag, spec_phase, fs, window_size, step_size):
    mag = np.absolute(spec_mag)
    if spec_phase is None:
        phase = np.random.randn(*spec_mag.shape)
    else:
        phase = np.exp(1.0j*np.unwrap(np.angle(spec_phase)))
    duration = (spec_mag.shape[0]-1)*step_size+window_size
    return istft(mag * phase,
                 fs,
                 duration,
                 step_size,
                 two_sided=False)
