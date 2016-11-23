import numpy as np
import scipy.io.wavfile as wav
import scikits.audiolab
import sys, glob


def padsig(sig, delay):
    '''
    Pad the signal at the end with a delay
    '''
    return np.append(sig[delay:], np.zeros(delay) )

def revsig(sig, delay):
    '''
    Reverse the delay given a single.
    '''
    if delay == 0:
        return sig
    return np.append(np.zeros(delay), sig[:-delay])

def buildrecv(sigs, delmatrix):
    '''
    Based on the delay matrix, specifying the signal received at each receiver. 
    For each "true" signal, we sum them at the receiver with specified offsets.
    '''
    if len(sigs) != len(delmatrix.T):
        print "ERROR: Signals not aligned with delay array"
        return -1
    recvm = []
    for delarray in delmatrix:
        recvv = []
        for j,sig in enumerate(sigs):
            recvv += [padsig( sig, delarray[j])]
        recvm += [recvv]
    return recvm

def sumrecv(sigs):
    '''
    With an list of list of true signals, this will simply add them together to
    show what the microphone would receive
    '''
    sumsigs = []
    for sig in sigs:
        sumsigs += [np.array(sig).sum(axis=0)]
    sumsigs = np.array(sumsigs)
    return (sumsigs.T / sumsigs.max(axis=1)).T

def beamform(sigs, sigindex, delmatrix):
    '''
    NOT IMPLEMENTED YET
    '''
    if len(sigs) != len(delmatrix.T):
        print "ERROR: Signals not aligned with delay array"
        return -1
    delarray = delmatrix.T[sigindex]
    recovered = np.zeros(len(sigs[0]))
    for i,sig in enumerate(sigs):
        recovered += revsig(sig, delarray[i]) 
    recovered /= recovered.max() 

    return recovered

if not len(sys.argv)==2 and not len(sys.argv)==3:
    sys.stdout.write("Usage: beamform <prefix> <speaker-id>\n")
    sys.exit(0)

sigprefix = sys.argv[1]
wavfiles = glob.glob(sigprefix+'*')

if len(sys.argv)==3:
    speakerid = int(sys.argv[2])
else:
    speakerid = np.random.randint(len(wavfiles))

# Read the wav files
allsigs = []
for i,wavfile in enumerate(wavfiles):
    (rate,sig) = wav.read(sigprefix+str(i)+'.wav')
    allsigs += [sig[:,0]]
numsigs = len(allsigs)
if not numsigs:
    sys.stdout.write("No files with prefix "+sigprefix+'\n')
    sys.exit(0)

# Generate the truth signals
delmatrix = np.random.randint( 0, 1e4, size=(numsigs,numsigs)  )
np.fill_diagonal( delmatrix, 0 )

# Build the signals as heard from the receiver
recsigs = buildrecv( allsigs, delmatrix )
sumsigs = sumrecv( recsigs )

# Beamform based on known delay matrices
beamformed = beamform( sumsigs, speakerid, delmatrix )

# Play the output
scikits.audiolab.play(sumsigs[2], fs=rate)
scikits.audiolab.play(beamformed, fs=rate)
# scikits.audiolab.play(recvall, fs=rate)

# For processing purposes
# import scipy.io
# scipy.io.wavfile.write('singlemic.wav', rate, sumsigs[2])
# scipy.io.wavfile.write('beamformed.wav', rate, beamformed)
