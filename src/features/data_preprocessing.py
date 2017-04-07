'''
Tools to preprocess audio data into datasets of  spectral features
'''

import os
import h5py
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from python_speech_features.sigproc import preemphasis
from .spectral_features import stft, istft

def undo_preemphasis(preemphasized_signal,coeff=0.95):
    """
    Function to undo the preemphasis of an input signal. The preemphasised
    signal p is computed from the signal s by the relation
                    p(n) = s(n) - coeff*s(n-1)
    with p(0) = s(0).  The inverse operation constructs the signal from the
    preemphasized signal with the recursion relation
                    s(n) = p(n) + coeff*s(n-1)

    Inputs:
        preemphasized_signal:  numpy array containing preemphasised signal
        coeff:   coefficient used to compute the preemphasized signal

    Returns:
        signal: numpy array containing the signal without preemphasis
    """

    # Get the length of the input and preallocate the output array
    length = preemphasized_signal.shape[0]
    signal = np.zeros(length)

    # Set the initial element of the signal
    signal[0] = preemphasized_signal[0]

    # Use the recursion relation to compute the output signal
    for i in range(1,length):
        signal[i] = preemphasized_signal[i] + coeff*signal[i-1]

    return signal

def make_stft_features(signal, sample_rate,
                       output_sample_rate=1e4,
                       window_size=0.0512, overlap=0.0256,
                       preemphasis_coeff=0.95):
    '''
    Function to take in a signal, resample it to output_sample_rate,
    normalize it, and compute the magnitude spectrogram.

    Inputs:
        signal: 1D numpy array containing signal to featurize (np.ndarray)
        sample_rate: sampling rate of signal (int)
        output_sample_rate: sample rate of signal after resampling (int)
        window_size: length of stft window in seconds (float)
        overlap: amount of overlap for stft windows (float)
        preemphasis: preemphasis coefficient (float)

    Returns:
        spectrogram: 2D numpy array with (Time, Frequency) components of
                     input signals (np.ndarray)
    '''

    # Downsample the signal to output_sample_rate
    resampled = resample_poly(signal,100,
                              int(sample_rate/output_sample_rate*100))

    # Do preemphasis on the resampled signal
    preemphasised = preemphasis(resampled,preemphasis_coeff)

    # Normalize the downsampled signal
    normalized = (preemphasised - preemphasised.mean())/preemphasised.std()

    # Get the magnitude spectrogram
    spectrogram = stft(normalized,output_sample_rate,
                       window_size,overlap,two_sided=False)

    return spectrogram

def make_stft_dataset(data_dir, key_level, file_type, output_file,
                      output_sample_rate=1e4,
                      window_size=0.0512, overlap=0.0256,
                      preemphasis_coeff=0.95,
                      track=None):
    '''
    Function to walk through a data directory data_dir and compute the stft
    features for each file of type file_type.  The computed features are
    then stored in an hdf5 file with name output_file

    Inputs:
        data_dir: directory containing data (possibly in subdirs) (str)
        key_level: Use folders at this depth as keys in the hdf5 file (int)
        file_type: Extension of data files (Ex: '.wav') (str)
        output_sample_rate: Sample rate to resample audio to (int)
        window_size: Length of fft window in seconds (float)
        overlap: Amount of window overlap in seconds (float)
        preemphasis_coeff: preemphasis coefficient (float)
        track: Track number to use for signals with multiple tracks (int)
    '''

    # Open output file for writing
    with h5py.File(output_file,'w') as data_file:

        # Walk through data_dir and process all the files
        for (dirpath, dirnames, filenames) in os.walk(data_dir, topdown=True):

            # Get the key corresponding to the directory name at depth equal
            # to key_level
            depth = dirpath[len(data_dir):].count(os.path.sep)
            if depth == key_level:
                _, key = os.path.split(dirpath)

                # Create a group for key if there isn't one already
                if key not in data_file:
                    data_file.create_group(key)

            # Process any files in this directory
            for file in filenames:
                if os.path.splitext(file)[1] == file_type:
                    file_path = os.path.join(dirpath,file)

                    # Read in the signal and sample rate
                    if track is not None:
                        signal, sample_rate = sf.read(file_path)
                        signal = signal[:,track]
                    else:
                        signal, sample_rate = sf.read(file_path)

                    # Compute STFT spectrogram
                    spectrogram = make_stft_features(signal,sample_rate,
                                                     output_sample_rate,
                                                     window_size,overlap,
                                                     preemphasis_coeff)

                    # Convert to 32 bit floats
                    spectrogram = spectrogram.astype(np.complex64)

                    data_file[key].create_dataset(os.path.splitext(file)[0],
                                                  data=spectrogram,
                                                  compression="gzip",
                                                  compression_opts=0)

