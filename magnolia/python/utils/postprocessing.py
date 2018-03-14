import numpy as np
from magnolia.preprocessing.preprocessing import undo_preemphasis
from magnolia.preprocessing.spectral_features import istft


def convert_preprocessing_parameters(params):
    p = params.copy()
    if 'n_fft' in p:
        del p['n_fft']
    if 'pad_mode' in p:
        del p['pad_mode']
    return p


def reconstruct(spec_mag, spec_full, fs, window_size, step_size, square=False,
    preemphasis=None):
    '''
    Reconstruct time-frequency signal from components representing
    magnitude and phase.

    Args:
        spec_mag: potentially complex time-frequency representation whose
            magnitude will be used in the reconstruction
        spec_full: complex time-frequency representation, whose phase
            will be used in reconstruction
        fs: sampling frequency
        window_size: not used
        step_size: step size between STFT frames
        square: take the square of the magnitude if True
        preemphasis: if not None, float: undo preemphasis at this level
    '''
    mag = np.absolute(spec_mag)
    if spec_full is None:
        phase = np.random.randn(*spec_mag.shape)
    else:
        phase = np.exp(1.0j*np.unwrap(np.angle(spec_full)))

    if square:
        mag = mag ** 2

    signal = istft(mag * phase,
                 fs,
                 None,
                 step_size,
                 two_sided=False)

    if preemphasis is not None:
        signal = undo_preemphasis(signal, preemphasis)

    return signal
