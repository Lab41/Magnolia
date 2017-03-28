"""wav_iterator.py

Tools to load WAV files representing sources, randomly mix them together,
and featurize them using stft, then iterate over the results

"""

import os
from itertools import islice
import numpy as np
import python_speech_features as psf
from scipy.io import wavfile

from . import spectral_features

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
