from itertools import combinations
import numpy as np
from scipy.signal import spectrogram

def compare_signals(sigs, fft_size=512):
    # Fcomp from original src
    spectrograms = np.stack([spectrogram(x, 1.0, 'hanning', fft_size, 0)[2] for x in sigs], axis=2)
    # Fn from original src
    magnitude_resps = np.sum(spectrograms * np.conj(spectrograms), axis=1)
    
    sims = np.zeros((sigs.shape[0], sigs.shape[0]))
    for idx_a, idx_b in combinations(range(sigs.shape[0]), 2):
        coherence = np.sum(spectrograms[:,:,idx_a] * np.conj(spectrograms[:,:,idx_b]), axis=1)
        coherence = (coherence * np.conj(coherence)) / magnitude_resps[:,idx_a]
        coherence = coherence / magnitude_resps[:,idx_b]
        sims[idx_a,idx_b] = sims[idx_b, idx_a] = np.mean(coherence)
    return sims
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    a = np.linspace(0, 3, 15000)
    sigs = np.vstack((
        np.sin(a),
        np.sin(np.sin(a)**3),
        np.sin(a+0.05)
    ))
    sims = compare_signals(sigs)
    plt.imshow(sims,interpolation='nearest')