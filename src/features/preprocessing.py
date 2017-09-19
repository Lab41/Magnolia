"""Tools to preprocess audio data into datasets of spectral features"""

import os
import logging
import h5py
import numpy as np
# import soundfile as sf
from tqdm import tqdm
from scipy.signal import resample_poly
# from python_speech_features.sigproc import preemphasis
import librosa as lr
from .spectral_features import stft, istft


logger = logging.getLogger('preprocessing')


def normalize_waveform(y):
    y = y - np.mean(y)
    return y/np.std(y)


def preemphasis(signal, coeff=0.95):
    """Perform preemphasis on the input signal.
    Inputs:
        signal: The signal to filter.
        coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    Returns:
        the filtered signal.

    This is taken directly from:
    https://github.com/jameslyons/python_speech_features/blob/e51df9e484da6c52d30b043f735f92580b9134b4/python_speech_features/sigproc.py#L133
    """

    if coeff == 0.0:
        return signal

    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def undo_preemphasis(preemphasized_signal, coeff=0.95):
    """Undo the preemphasis of an input signal. The preemphasised
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

    if coeff == 0.0:
        return preemphasized_signal

    # Get the length of the input and preallocate the output array
    length = preemphasized_signal.shape[0]
    signal = np.zeros(length)

    # Set the initial element of the signal
    signal[0] = preemphasized_signal[0]

    # Use the recursion relation to compute the output signal
    for i in range(1, length):
        signal[i] = preemphasized_signal[i] + coeff*signal[i-1]

    return signal


def preprocess_waveform(y, sample_rate,
                        target_sample_rate,
                        preemphasis_coeff=0.95,
                        stft_args={}):
    """Resample, preemphasize, and compute the magnitude spectrogram.
    Inputs:
        y: 1D numpy array containing signal to featurize (np.ndarray)
        sample_rate: sampling rate of signal (int)
        target_sample_rate: sample rate of signal after resampling (int)
        preemphasis_coeff: preemphasis coefficient (float)
        stft_args: keywords args for librosa's stft function

    Returns:
        D: 2D numpy array with (stft_args['n_fft']/2 + 1, Time) components of
           input signals (np.ndarray)
        n: length of waveform for inverse transform
    """

    # Resample the signal to target_sample_rate
    if sample_rate != target_sample_rate:
        y = lr.core.resample(y, sample_rate, target_sample_rate)

    # Do preemphasis on the resampled signal
    y = preemphasis(y, preemphasis_coeff)

    # Perform stft with librosa
    n = len(y)
    n_fft = 2048
    if 'n_fft' in stft_args:
        n_fft = stft_args['n_fft']
    y_pad = lr.util.fix_length(y, n + n_fft // 2)
    D = lr.core.stft(y_pad, **stft_args)

    return D, y


def undo_preprocessing(D, n,
                       preemphasis_coeff=0.95,
                       istft_args={}):
    """Undo the operations of the preprocessing (aside from resampling).
    Inputs:
        D: 2D numpy array of spectrogram (np.ndarray)
        n: length of original waveform (int)
        preemphasis_coeff: preemphasis coefficient (float)
        istft_args: keywords args for librosa's istft function

    Returns:
        y: 1D numpy waveform array (np.ndarray)
    """

    # Undo stft with librosa
    y = lr.core.istft(D, length=n, **istft_args)

    # Undo preemphasis on the signal
    y = undo_preemphasis(y, preemphasis_coeff)

    return y


def make_stft_features(signal, sample_rate,
                       output_sample_rate=10000,
                       window_size=0.0512, overlap=0.0256,
                       preemphasis_coeff=0.95, fft_size=512):
    """Take a signal, resample it to output_sample_rate,
    normalize it, and compute the magnitude spectrogram.
    Inputs:
        signal: 1D numpy array containing signal to featurize (np.ndarray)
        sample_rate: sampling rate of signal (int)
        output_sample_rate: sample rate of signal after resampling (int)
        window_size: length of stft window in seconds (float)
        overlap: amount of overlap for stft windows (float)
        preemphasis: preemphasis coefficient (float)
        fft_size: length (in seconds) of DFT window (float)

    Returns:
        spectrogram: 2D numpy array with (Time, Frequency) components of
                     input signals (np.ndarray)
    """

    # Downsample the signal to output_sample_rate
    resampled = resample_poly(signal, 100,
                              int(sample_rate/float(output_sample_rate)*100))

    # Do preemphasis on the resampled signal
    preemphasised = preemphasis(resampled, preemphasis_coeff)

    # Normalize the downsampled signal
    normalized = (preemphasised - preemphasised.mean())/preemphasised.std()

    # Get the spectrogram
    spectrogram = stft(normalized, output_sample_rate,
                       window_size, overlap, two_sided=False, fft_size=fft_size)

    return spectrogram


def undo_stft_features_old(spectrogram, sample_rate=10000,
                       window_size=0.0512,
                       preemphasis_coeff=0.95, fft_size=512):
    """Undo the preprocessing operations
    Inputs:
        signal: 2D preprocessed spectrogram (np.ndarray)
        sample_rate: sampling rate of signal (int)
        window_size: length of stft window in seconds (float)
        preemphasis: preemphasis coefficient (float)
        fft_size: length (in seconds) of DFT window (float)

    Returns:
        waveform: (np.ndarray)
    """

    # Invert the stft operation
    x = istft(spectrogram, sample_rate, None, window_size,
              two_sided=False, fft_size=fft_size)

    # Undo preemphasis
    x = undo_preemphasis(x, preemphasis_coeff)

    return x



def make_stft_dataset(data_dir, spectrogram_output_file,
                      waveform_output_file,
                      compression="gzip",
                      compression_opts=0,
                      key_maker=None,
                      **kwargs):
    """Walk through a data directory data_dir and compute the stft
    features for each file of type file_type. The computed features are
    then stored in an hdf5 file with name output_file
    Inputs:
        data_dir: directory containing data (possibly in subdirs) (str)
        spectrogram_output_file: HDF5 output file name that stores the preprocessed spectrograms
        waveform_output_file: HDF5 output file name that stores the preprocessed waveforms
        compression: compression algorithm for h5py to use when storing data
        compression_opts: compression level (0 = no compression)
        key_maker: object that when given a file name will return a key (object)
    Output:
        HDF5 file
    """

    # Open output file for writing
    spectrogram_data_file = h5py.File(spectrogram_output_file, 'w')
    waveform_data_file = h5py.File(waveform_output_file, 'w')

    # Walk through data_dir and process all the files
    filenames = lr.util.files.find_files(data_dir)
    logger.info('starting loop over data')
    for filename in tqdm(filenames):

        # Read in the signal and sample rate
        if 'track' in kwargs:
            signal, sample_rate = lr.core.load(filename, sr=None, mono=False)
            signal = signal[kwargs['track']]
            del kwargs['track']
        else:
            signal, sample_rate = lr.core.load(filename, sr=None, mono=True)

        # Compute STFT spectrogram
        signal = normalize_waveform(signal)
        D, y = preprocess_waveform(signal, sample_rate, **kwargs)

        # Check that spectrogram has sufficient signal
        if np.allclose(np.abs(D), np.zeros_like(D)):
            continue

        # Determine key
        key = None
        dataset_name = None
        if key_maker is not None:
            key, dataset_name = key_maker.process_file_metadata(filename,
                                                                y, D, **kwargs)
            if key not in spectrogram_data_file:
                logger.debug('creating key {}'.format(key))
                spectrogram_data_file.create_group(key)
                waveform_data_file.create_group(key)

        spec_ds = None
        wf_ds = None
        if key is not None:
            spec_ds = spectrogram_data_file[key].create_dataset(dataset_name,
                                               data=D,
                                               compression=compression,
                                               compression_opts=compression_opts)
            wf_ds = waveform_data_file[key].create_dataset(dataset_name,
                                               data=y,
                                               compression=compression,
                                               compression_opts=compression_opts)
        else:
            spec_ds = spectrogram_data_file.create_dataset(os.path.splitext(filename)[0],
                                          data=D,
                                          compression=compression,
                                          compression_opts=compression_opts)
            wf_ds = waveform_data_file.create_dataset(os.path.splitext(filename)[0],
                                          data=y,
                                          compression=compression,
                                          compression_opts=compression_opts)
        wf_ds.attrs['spectral_length'] = D.shape[1]

    logger.info('looping over data finished')


def make_stft_dataset_old(data_dir, key_level, file_type, output_file,
                          output_sample_rate=1e4,
                          window_size=0.0512, overlap=0.0256,
                          preemphasis_coeff=0.95, fft_size=512,
                          track=None):
    '''DEPRECATED!
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
        fft_size: length (in seconds) of DFT window (float)
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
                                                     preemphasis_coeff,
                                                     fft_size=fft_size)

                    # Convert to 32 bit floats
                    spectrogram = spectrogram.astype(np.complex64)

                    data_file[key].create_dataset(os.path.splitext(file)[0],
                                                  data=spectrogram,
                                                  compression="gzip",
                                                  compression_opts=0)
