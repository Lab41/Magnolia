'''
Tools to preprocess audio data into datasets of  spectral features
'''

import os
import h5py
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import stft, istft

def make_stft_features(signal, sample_rate, output_sample_rate=1e4,
                       window_size=0.05, overlap=0.025):
    '''
    Function to take in a signal, resample it to output_sample_rate, 
    normalize it, and compute the magnitude spectrogram. 

    Inputs:
        signal: 1D numpy array containing signal to featurize (np.ndarray)
        sample_rate: sampling rate of signal (int)
        output_sample_rate: sample rate of signal after resampling (int)
        window_size: length of stft window in seconds (float)
        overlap: amount of overlap for stft windows (float)

    Returns:
        spectrogram: 2D numpy array with (Time, Frequency) components of
                     input signals (np.ndarray)
    '''

    # Downsample the signal to output_sample_rate
    resampled = resample_poly(signal,100,
                              int(sample_rate/output_sample_rate*100))
    
    # Normalize the downsampled signal
    resampled = (resampled - resampled.mean())/resampled.std()
    
    # Get the magnitude spectrogram
    spectrogram = stft(resampled,sample_rate,
                       window_size,overlap,two_sided=False)

    return spectrogram
