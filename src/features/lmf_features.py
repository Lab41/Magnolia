import os
from itertools import islice
import numpy as np
import scipy.signal
import python_speech_features as psf
from . import spectral_features

class LmfIterator:
    def __init__(self, spectrograms, transform_which=None,
                sample_rate=10000, num_filters=40,
                diff_features=False):
        ''' Iterator over log mel-frequency features, given an iterator over STFT
        features.

        Args:
            spectrograms (iterable): should yield an iterable
              (e.g. a tuple) of STFT features, each one of shape
              (time x frequency [x ...])
            transform_which (tuple or None): if not None, which of the STFT
              features yielded from spectrograms should be transformed?
              For example, if spectrograms yields a 3-tuple of STFT features,
              `transform_which=(0,)` means that only the first member of that
              tuple is transformed
            sample_rate (float): sample rate of the signals represented in
              spectrograms
            num_filters (int): number of filters to generate; determines
              output dimension of log mel-frequency features
            diff_features (bool): If true, generate first- and second-order
              difference features in time

        Yields:
            tuple, same shape as the elements of `spectrograms`, with one or more
              feature sets transformed into leg mel-frequency filterbank
              features
        '''
        self.sample_rate = sample_rate
        self.num_fft = None
        self.num_filters = num_filters
        self.spectrograms = spectrograms
        self.transform_which = transform_which
        self.diff_features = diff_features

    def __next__(self):
        spectrograms = next(self.spectrograms)
        # Initialize number of bins
        if self.num_fft is None:
            self.num_fft = 2*(spectrograms[0].shape[1]-1)
        transformed_spectrograms = []
        for i, spectrogram in enumerate(spectrograms):
            if self.transform_which is None or i in self.transform_which:
                transformed_spectrograms.append(self._transform(spectrogram))
            else:
                transformed_spectrograms.append(spectrogram)
        return tuple(transformed_spectrograms)

    def __iter__(self):
        return self

    def _transform(self, spectrogram):
        '''Transform STFT features into log mel-frequency filterbank features'''
        # handle multiple spectrograms at once
        spec_dim = len(spectrogram.shape)
        if spec_dim > 2:
            new_shape = [*range(2,spec_dim), 0, 1]
            print(new_shape)
            spectrogram = np.transpose(spectrogram, new_shape)
        # freely adapted from python_speech_features logfbank
        fb = psf.get_filterbanks(self.num_filters,
                                self.num_fft,
                                self.sample_rate,0,
                                self.sample_rate/2)
        mag_spec = np.absolute(spectrogram)
        pow_spec = 1.0/self.num_fft * np.square(mag_spec)
        energies = np.dot(mag_spec,fb.T) # compute the filterbank energies
        energies = np.where(energies == 0,np.finfo(float).eps,energies)
        log_energies = np.log(energies)
        if spec_dim > 2:
            old_shape = [spec_dim-2, spec_dim-1, *list(range(spec_dim-2))]
            print(old_shape)
            log_energies = np.transpose(log_energies, old_shape)

        # get diff features
        if self.diff_features:
            first_diff = np.diff(log_energies, 1, axis=0)
            # Zero-pad beginning
            first_diff = np.concatenate((np.zeros((1,*first_diff.shape[1:]),
                                                 dtype=log_energies.dtype),
                                                 first_diff), axis=0)
            second_diff = np.diff(log_energies, 2, axis=0)
            second_diff = np.concatenate((np.zeros((2,*second_diff.shape[1:]),
                                                 dtype=log_energies.dtype),
                                                 second_diff), axis=0)
            # Concatenate in frequency
            # This is a weird thing to do, not sure if would be better to
            # have on its own dimension
            log_energies = np.concatenate((log_energies, first_diff, second_diff),
                                          axis=1)
        return log_energies


def lmf_stft_iterator(wavs, fs = 1.0, stft_len=1024, stft_step=512, nfft=512,
    nfilters=40, use_diffs=True, mode='complex', **kwargs):
    """
    get signals from iterator wavs, transform to freq domn,
    get difference features, and yield
    Warning: stft length should make sense in reln. to length of signals yielded by wavs

    Args:

      **kwargs - passed to stft

    Yields:
        tuple of NDArrays - (features, truth) where
            features is time x freq and
            truth is num_srcs + 1 x time x freq, with the mixed signal's STFT
            in the last slot on the first axis

    """

    while True:
        truth_sigs, mixed_sig = next(wavs)
        # Stack signals onto each other for transformation
        all_sigs = (truth_sigs, mixed_sig)
        all_sigs = np.concatenate(all_sigs, axis=0)
        # transform each truth signal into logmel features
        lmfs = []
        stfts = []

        for sig in all_sigs:
            stft = scipy.signal.spectrogram(sig, fs=fs, nperseg=stft_len,
                                            noverlap=stft_len-stft_step,
                                            mode=mode, scaling='density', **kwargs)[2].T
            stfts.append(stft)

        stft = np.stack(stfts, 0)

        lmf = psf.logfbank(mixed_sig, samplerate=fs,
                nfft=nfft, nfilt=nfilters,
                winlen=stft_len, winstep=stft_step)

        if use_diffs:
            # take 1st- and 2nd-order differences of LMF in time
            diff1 = np.diff(lmf, axis=0)
            diff2 = np.diff(diff1, axis=0)
            # zero-pad diff features
            diff1 = np.concatenate((np.zeros_like(diff1[:1]), diff1))
            diff2 = np.concatenate((np.zeros_like(diff2[:2]), diff2))
            # concatenate difference features in "frequency" TODO: use another dimension??
            lmf = np.concatenate((lmf, diff1, diff2), axis=1)

        yield lmf, stft
