from scipy.io import wavfile
import numpy as np

def norm_wav( wave ):

    if type(wave)== str:
      fs, wav = wavfile.read( wave )
    else:
      wav = wave

    wav = wav.astype( np.float32 )
    wav -= wav.mean() 
    wav /= wav.std()

    return wav
   
