"""wav_iterator.py

Tools to load WAV files representing sources, randomly mix them together,
and featurize them using stft, then iterate over the results

"""

import os
import numpy as np
import python_speech_features as psf
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
    return fs, truth, mixed

def wav_iterator(batch_size, wav_dir, stft_len=1024.0, stft_step_ratio=0.5, nfft=512, nfilters=40, **kwargs):
    """
    get signals, transform to freq domn, get difference features, and batch out
    Warning: stft settings should match sig_length parameter to wav_mixer
    
    Returns:
    truth_lmf - num_srcs x (num_time_steps-2) x (3*num_freq_bins)
    mixed_lmf - (num_time_steps-2) x (3*num_freq_bins)
    
    """
    stft_len_orig = stft_len
    for i in range(batch_size):
        fs, truth_sig, mixed_sig = wav_mixer(wav_dir, **kwargs)
        stft_len = (stft_len_orig)/fs
        # transform into logmel features
        truth_lmf = np.stack([psf.logfbank(truth_sig[j], samplerate=fs, nfft=nfft, nfilt=nfilters, 
                       winlen=stft_len, winstep=stft_len * stft_step_ratio) for j in 
                              range(truth_sig.shape[0])], axis=-1)
        
        truth_lmf = np.transpose(truth_lmf, [2, 0, 1])
        diff1 = np.diff(truth_lmf, axis=1)
        diff2 = np.diff(diff1, axis=1)
        
        truth_lmf = np.concatenate((truth_lmf[:,:-2], diff1[:,:-1], diff2), axis=2)
        mixed_lmf = psf.logfbank(mixed_sig, samplerate=fs, nfft=nfft, nfilt=nfilters, 
                       winlen=stft_len, winstep=stft_len * stft_step_ratio)
        
        diff1 = np.diff(mixed_lmf, axis=0)
        diff2 = np.diff(diff1, axis=0)
        mixed_lmf = np.concatenate((mixed_lmf[:-2], diff1[:-1], diff2), axis=1)
        
        yield truth_lmf, mixed_lmf

def wav_batches(batch_size, wav_dir, **kwargs):
    """
    Yield batches as numpy arrays
    
    Yields:
    truth - batch_size x num_srcs x num_time_steps x num_freq_bins
    mix - batch_size x num_time_steps x num_freq_bins
    """
    while True:
        truth_tensors, mix_tensors = list(zip(*wav_iterator(batch_size, wav_dir, **kwargs)))
        yield np.stack(truth_tensors), np.stack(mix_tensors)
   