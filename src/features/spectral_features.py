''' spectral_features.py

Functions for featurizing audio using spectral analysis, as well as for
reconstructing time-domain audio signals from spectral features
'''

import sys
import logging
import numpy as np
import scipy
import scipy.signal

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
        fft_size - number of DFT points

    Output:
        X = 2d array time-frequency repr of x, time x frequency
    '''

    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    overlap_samp = framesamp - hopsamp

    _, _, X = scipy.signal.stft(x, fs, window='hann', nperseg=framesamp,
        noverlap=overlap_samp, nfft=fft_size, return_onesided=not two_sided)
    return X.T

def istft(X, fs, recon_size, hop, two_sided=True, fft_size=None):
    ''' Inverse Short Time Fourier Transform (iSTFT) - Spectral reconstruction

    Input:
        X - set of 1D time-windowed spectra, time x frequency
        fs - sampling frequency (in Hz)
        recon_size - Not used
        hop - skip rate between successive windows
        fft_size - number of DFT points

    Output:
        x - a 1-D array holding reconstructed time-domain audio signal
    '''
    if two_sided:
        framesamp = X.shape[1]
    else:
        framesamp = 2*(X.shape[1] - 1)
    hopsamp = int(hop*fs)
    overlap_samp = framesamp - hopsamp


    _, x = scipy.signal.istft(X.T, fs, window='hann', nperseg=framesamp,
        nfft=fft_size, noverlap = overlap_samp,
        input_onesided=not two_sided)
    if recon_size is not None and recon_size != x.shape[0]:
        logger = logging.getLogger(__name__)
        logger.warn("Size of reconstruction ({}) does not match value of "
        "deprecated recon_size parameter ({}).".format(x.shape[0], recon_size))
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

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG)
    # from scipy.signal import spectrogram
    np.set_printoptions(precision=3,suppress=True)

    # Try wrapper functions
    try:
        sig = np.array([0,1,0.5,0.1]*10)
        sig_stft = stft(sig, 1, 8, 4, two_sided=False)
        print(sig_stft)
        print(sig_stft.shape)
        sig_recon = istft(sig_stft, 1, 47, 4, two_sided=False)
        print(sig_recon)
        print(sig_recon.shape)
    except:
        print("Trouble 1")
        print(sys.exc_info())
        raise

    # Cf. scipy.signal
    try:
        sig = np.array([0,1,0.5,0.1]*10)
        _, _, sig_stft = scipy.signal.stft(sig, 1, 'hann', 8, 4, return_onesided=True)
        print(sig_stft)
        _, sig_recon = scipy.signal.istft(sig_stft, 1, 'hann', 8, 4, input_onesided=True)
        print(sig_recon)
    except:
        print("Trouble 2")
        print(sys.exc_info())
        raise
