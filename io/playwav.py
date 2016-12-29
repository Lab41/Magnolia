import numpy as np
import scipy.io.wavfile as wav
import scikits.audiolab
import sys, glob


if not len(sys.argv)==2:
    sys.stdout.write("Usage: python playwav.py <filename> \n")
    sys.exit(0)

# Read the wav files
(rate,sig) = wav.read(sys.argv[1])
if len(sig.shape)==2:
    sig =sig[:,0].astype(np.float64)/sig[:,0].max()

scikits.audiolab.play(sig, fs=rate)
