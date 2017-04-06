from python_speech_features import sigproc
from keras.models import load_model
from python_speech_features.sigproc import deframesig
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np

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

def signal_reconstruction(mask,mag,phase,fs=16000,winlen=0.01,winstep=0.005):

    recon_signal = (mask * mag) * np.exp( 1j *  phase)
    recon_signal = np.fft.irfft(recon_signal)
    recon_signal = recon_signal[:,:(int(fs*winlen))]
    recon_signal = deframesig(recon_signal, 0, int(fs*winlen), int(fs*winstep))

    return recon_signal


def keras_separate(signal_path, model_path):
    
    """
    Reads in the signal from signal_path and uses the model to separate it into
    component sources.
    Inputs:
        signal_path:  Path to audio file containing the signal
        model: Model object with a predict method to separate signals

    Outputs:
        sources: list of separated speaker numpy arrays
    """

    nfilt=64
    numcep=64
    nfft=512
    winlen=0.01
    winstep=0.005
    ceplifter=0
    fs = 16000

    wav_list = []

    fs,noisy_signal = wav.read(signal_path)

    mfcc_feat = mfcc(noisy_signal,fs,nfilt=nfilt,numcep=numcep,nfft=nfft,
                 winlen=winlen,winstep=winstep,ceplifter=ceplifter,
                 appendEnergy=False)

    mfcc_magni = specdecomp(noisy_signal,samplerate=fs,nfft=nfft,
                        winlen=winlen,winstep=winstep,decomp='abs')

    mfcc_phase = specdecomp(noisy_signal,samplerate=fs,nfft=nfft,
                        winlen=winlen,winstep=winstep,decomp='phase')

    model = load_model(model_path)

    mask_1 = model.predict(mfcc_feat)
    mask_2 = 1 - mask_1

    mask_list = [mask_1,mask_2]

    for mask in mask_list:
        recon_signal = (signal_reconstruction(mask,mfcc_magni,mfcc_phase)).astype(np.int16)
        wav_list.append(recon_signal)

    return wav_list

def keras_spec(signal_path):

    nfilt=64
    numcep=64
    nfft=512
    winlen=0.01
    winstep=0.005
    ceplifter=0
    fs = 16000

    fs,noisy_signal = wav.read(signal_path)

    mfcc_feat = mfcc(noisy_signal,fs,nfilt=nfilt,numcep=numcep,nfft=nfft,
                 winlen=winlen,winstep=winstep,ceplifter=ceplifter,
                 appendEnergy=False)

    return mfcc_feat







