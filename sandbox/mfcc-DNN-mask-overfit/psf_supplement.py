import numpy as np
from python_speech_features import sigproc

def specdecomp(signal,samplerate=16000,winlen=0.025,winstep=0.01,
              nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
              winfunc=lambda x:np.ones((x,)),decomp='complex'):

    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1). 
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """    
    
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    if decomp=='time' or decomp=='frames':
        return frames
    
    complex_spec = np.fft.rfft(frames,nfft)    
    if decomp=='magnitude' or decomp=='mag' or decomp=='abs':
        return np.abs(complex_spec)
    elif decomp=='phase' or decomp=='angle':
        return np.angle(complex_spec)
    elif decomp=='power' or decomp=='powspec':
        return sigproc.powspec(frames,nfft)
    else:
        return complex_spec        
    return spect