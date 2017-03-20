"""wav_iterator.py

Tools to load WAV files representing sources, randomly mix them together,
and featurize them using stft, then iterate over the results

"""

import os
from itertools import islice
import numpy as np
import python_speech_features as psf
from .spectral_features import stft
from scipy.io import wavfile

def wav_mixer(wav_dir, mix_random=False, num_to_mix=2, sig_length=100*512+1, mask="_src.wav", dtype=np.int16):
    """
    In any directory of wavs with all the same sampling freqs, pick some
    wavs, take random slices, and mix them together.

    Returns the original source signals and the mixed-down signal.
    """
    wav_candidates = [x for x in os.listdir(wav_dir) if mask in x]
    truth = np.zeros((num_to_mix, sig_length), dtype=dtype)
    selected_candidates = np.random.choice(wav_candidates, num_to_mix, replace=False)

    for i, wav_name in enumerate(selected_candidates):
        fs, wav = wavfile.read(os.path.join(wav_dir, wav_name))
        start = np.random.randint(0, high=wav.shape[0]-sig_length)

        end = start + sig_length
        truth[i, :] = wav[start:end]

    if mix_random:
        mixing_system = np.random.rand(1, num_to_mix)
    else:
        mixing_system = np.ones((1, num_to_mix))

    mixed = mixing_system @ truth
    return truth, mixed

def wav_iterator(wav_dir, **kwargs):
    while True:
        yield wav_mixer(wav_dir, **kwargs)

def batcher(feature_iter, batch_size=256):
    '''
    Yield batches from an iterator over examples.
    batch_size examples from feature_iter will be collected from feature_iter
    and 'transposed' so that elements in a given position
    will be grouped together across examples.

    Yields:
        tuple, the same length as one item from feature_iter, of iterables
            of size batch_size
    '''
    while True:
        # Gather batch_size examples from feature_iter
        new_batch = islice(feature_iter, batch_size)
        # Transpose the batch so that examples of a certain type are
        # grouped together
        try:
            batch_transposed = []
            for dataset in list(zip(*new_batch)):
                try:
                    batch_transposed.append(np.array(dataset))
                except ValueError:
                    batch_transposed.append(tuple(dataset))
            yield tuple(batch_transposed)
        except ValueError:
            raise StopIteration

def test_batcher():
    # Basic functionality: transpose for non-array-like data; cast to array when possible
    features = [[2, 0], [5, 3], [8, 10], [2, -4]]
    batches = batcher(iter(features), 2)
    a, b = list(islice(batches, 2))
    assert (a[1] == np.array((0, 3))).all()
    assert (b[0] == np.array((8, 2))).all()

    # Fall back to tuple when shapes don't conform for a scertain dataset
    features = [[2, 0], [5, [0,2]], [8, 10], [2, -4]]
    batches = batcher(iter(features), 2)
    c = list(islice(batches, 1))
    print(c)
    assert(isinstance(c[0][0], np.ndarray))
    assert(isinstance(c[0][1], tuple))

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

def stft_iterator(wavs, fs = 1.0, stft_len=1024, stft_step=512, use_diffs=False, **kwargs):
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
        # transform each truth signal into stft features
        num_sigs = all_sigs.shape[0]
        # TODO: Is axis=-1 the only option here? Transpose in next line seems
        # unnecessary
        spectrogram = np.stack([stft(all_sigs[j], fs=fs, framesz=stft_len, hop=stft_step, **kwargs)
                                for j in range(num_sigs)], axis=-1)                            
        # From time x freq x sig, transform to sig x time x freq
        spectrogram = np.transpose(spectrogram, [2, 0, 1])
        if use_diffs:
            # take 1st- and 2nd-order differences in time
            diff1 = np.diff(spectrogram, axis=1)
            diff2 = np.diff(diff1, axis=1)
            # concatenate difference features in "frequency" TODO: use another dimension??
            spectrogram = np.concatenate((spectrogram[:,:-2], diff1[:,:-1], diff2), axis=2)

        truth_lmf = spectrogram[1:]
        mixed_lmf = spectrogram[0]

        yield truth_lmf, mixed_lmf


# def wav_batches(batch_size, wav_dir, **kwargs):
#     """
#     Yield batches as numpy arrays
#
#     Yields:
#     truth - batch_size x num_srcs x num_time_steps x num_freq_bins
#     mix - batch_size x num_time_steps x num_freq_bins
#     """
#     while True:
#         truth_tensors, mix_tensors = list(zip(*wav_iterator(batch_size, wav_dir, **kwargs)))
#         yield np.stack(truth_tensors), np.stack(mix_tensors)
