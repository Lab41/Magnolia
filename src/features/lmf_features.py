import os
from itertools import islice
import numpy as np
import scipy.signal
import python_speech_features as psf
from . import spectral_features

class LmfIterator:
    def __init__(self, spectrograms, transform_which=None,
                sample_rate=10000, num_filters=40,
                window_size=0.05, window_step=0.025,
                diff_features=False):
        self.sample_rate = sample_rate
        self.num_fft = None
        self.num_filters = num_filters
        self.window_size = window_size
        self.window_step = window_step
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


def lmf_iterator(wavs, fs = 1.0, stft_len=1024, stft_step=512, nfft=512,
    nfilters=40, use_diffs=True):
    """
    get signals from iterator wavs, transform to freq domn,
    get difference features, and yield
    Warning: stft settings should match sig_length parameter to wav_mixer

    Returns:
    truth_lmf - num_srcs x (num_time_steps-2) x (3*num_freq_bins)
    mixed_lmf - (num_time_steps-2) x (3*num_freq_bins)

    """

    #stft_len_orig = stft_len
    while True:
        truth_sigs, mixed_sig = next(wavs)
        # Stack signals onto each other for transformation
        all_sigs = (mixed_sig, truth_sigs)
        all_sigs = np.concatenate(all_sigs, axis=0)
        #stft_len = (stft_len_orig)/fs
        # transform each truth signal into logmel features
        #num_sigs = all_sigs.shape[0]
        lmfs = []
        for sig in all_sigs:
            lmf = psf.logfbank(sig, samplerate=fs,
                                nfft=nfft, nfilt=nfilters,
                                winlen=stft_len, winstep=stft_step)

            lmfs.append(lmf)
        lmf = np.stack(lmfs, 0)
        # From time x freq x sig, transform to sig x time x freq
        #lmf = np.transpose(lmf, [2, 0, 1])
        if use_diffs:
            # take 1st- and 2nd-order differences of LMF in time
            diff1 = np.diff(lmf, axis=0)
            diff2 = np.diff(diff1, axis=0)
            # zero-pad diff features
            diff1 = np.concatenate((np.zeros_like(diff1[:1]), diff1))
            diff2 = np.concatenate((np.zeros_like(diff2[:2]), diff2))
            # concatenate difference features in "frequency" TODO: use another dimension??
            lmf = np.concatenate((lmf, diff1, diff2), axis=1)

        truth_lmf = lmf[1:]
        mixed_lmf = lmf[0]

        yield truth_lmf, mixed_lmf


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
