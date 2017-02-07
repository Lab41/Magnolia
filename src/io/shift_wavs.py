import logging
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

def shift_signal(sig1, sig2):
    """
    Returns version of sig2 aligned to sig1 using the cross-correlation
    and truncated/zero-padded as necessary to give it the same shape as sig1
    """
    logger = logging.getLogger(__name__)
    # Get cross-correlation
    xc = fftconvolve(sig1, sig2[::-1], mode='full')
    xaxis=np.arange(-len(sig1)+1,len(sig2))
    logger.info(xaxis.shape)
    logger.info(xc.shape)
    # How far in to shift sig2
    offset = -1*int(np.argmax(xc)-sig2.size)
    shifted2=sig2.copy()
    # Shift signal start
    if offset >= 0:
        logger.info("Truncate head...{}".format(offset))
        shifted2 = sig2[offset:]
    else:
        logger.info("Padding head...{}".format(offset))
        shifted2 = np.concatenate((np.zeros(-offset, dtype=sig2.dtype), sig2))
    # Deal with signal tail
    if shifted2.size >= sig1.size:
        shifted2 = shifted2[:sig1.size]
    else:
        padding = np.zeros(sig1.size-shifted2.size, dtype=shifted2.dtype)
        shifted2 = np.concatenate((shifted2, padding))

    assert sig1.size==shifted2.size
    return shifted2

def shift_wavs(*wav_paths):
    wav_0 = None
    for wav_path in wav_paths:
        fs, sig = wavfile.read(wav_path)
        if wav_0 is None:
            wav_0 = sig
            yield wav_0
        else:
            yield shift_signal(wav_0, sig)
