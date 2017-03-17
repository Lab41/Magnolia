import numpy as np
import scipy.signal
import python_speech_features as psf
from . import spectral_features

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
