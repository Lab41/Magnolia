import sys
import time

import numpy as np
import tensorflow as tf

sys.path.append('../../')

from src.dnnseparate.L41model import L41Model

from src.utils.clustering_utils import clustering_separate, get_cluster_masks, process_signal

from src.features.mixer import FeatureMixer
from src.features.supervised_iterator import SupervisedIterator, SupervisedMixer
from src.features.hdf5_iterator import SplitsIterator
from src.features.spectral_features import istft
from src.features.data_preprocessing import undo_preemphasis
from src.utils.bss_eval import bss_eval_sources

import IPython
from IPython.display import display, Audio

fft_size = 512

numsources = 2
batchsize = 256
datashape = (40, 257)
embedding_size = 600
libridev='/local_data/teams/magnolia/librispeech/processed_dev-clean.h5'
libritrain='/local_data/teams/magnolia/librispeech/processed_train-clean-100.h5'
libritest='/local_data/teams/magnolia/librispeech/processed_test_clean.h5'

with open('../../data/librispeech/authors/train-clean-100-F.txt','r') as speakers:
    in_set_F = speakers.read().splitlines()
    
with open('../../data/librispeech/authors/train-clean-100-M.txt','r') as speakers:
    in_set_M = speakers.read().splitlines()

with open('../../data/librispeech/authors/test-clean-F.txt','r') as speakers:
    out_set_F = speakers.read().splitlines()
    
with open('../../data/librispeech/authors/test-clean-M.txt','r') as speakers:
    out_set_M = speakers.read().splitlines()    
    
in_set_all = in_set_F + in_set_M
out_set_all = out_set_F + out_set_M


# ### Create an instance of Lab41's model
model = L41Model(nonlinearity='tanh')
model.initialize()
# model.load('lab41_model-x-restart.ckpt')
model.load('lab41_nonorm-final.ckpt')

sample_rate = 1e4
overlap = 0.0256


def invert_spectrogram(magnitude,phase):
    return istft(np.square(magnitude)*np.exp(phase*1.0j),sample_rate,None,overlap,two_sided=False,fft_size=fft_size)

def bss_eval_sample(mixer, num_sources):
    data = next(mixer)
    
    mixes = [invert_spectrogram(np.abs(data[0]),np.unwrap(np.angle(data[0]))) for i in range(1,num_sources + 1)]
    sources = [invert_spectrogram(np.abs(data[i][1]),np.unwrap(np.angle(data[i][1]))) for i in range(1,num_sources + 1)]
    
    mixes = [undo_preemphasis(mix) for mix in mixes]
    sources = [undo_preemphasis(source) for source in sources]
    
    input_mix = np.stack(mixes)
    reference_sources = np.stack(sources)
    estimated_sources = clustering_separate(mixes[0],1e4,model,num_sources)
    
    do_nothing = bss_eval_sources(reference_sources, input_mix)
    do_something = bss_eval_sources(reference_sources, estimated_sources)
    
    sdr = do_something[0] - do_nothing[0]
    sir = do_something[1] - do_nothing[1]
    sar = do_something[2] - do_nothing[2]
    
    return {'SDR': sdr, 'SIR': sir, 'SAR': sar}

def bss_eval(mixer, num_sources, num_samples):
    SDR = np.zeros(num_samples)
    SIR = np.zeros(num_samples)
    SAR = np.zeros(num_samples)
    
    for i in range(num_samples):
        evals = bss_eval_sample(mixer, 2)
        SDR[i] = 1/(2)*(evals['SDR'][0] + evals['SDR'][1])
        SIR[i] = 1/(2)*(evals['SIR'][0] + evals['SIR'][1])
        SAR[i] = 1/(2)*(evals['SAR'][0] + evals['SAR'][1])
    
    return SDR, SIR, SAR


# ## Two Speaker In-Set Breakdown
# 
# Categories are:
# 1. MM
# 2. FF
# 3. MF
# 4. FM

maleiter = SplitsIterator([0.8,0.1,0.1], libritrain, speaker_keys=in_set_M, shape=(150,257), return_key=True)
maleiter.set_split(2)

femaleiter = SplitsIterator([0.8,0.1,0.1], libritrain, speaker_keys=in_set_F, shape=(150,257), return_key=True)
femaleiter.set_split(2)

samples = 500
MMmixer = SupervisedMixer([maleiter,maleiter], shape=(150,257), 
                          mix_method='add', diffseed=True)
FFmixer = SupervisedMixer([femaleiter,femaleiter], shape=(150,257), 
                          mix_method='add', diffseed=True)
MFmixer = SupervisedMixer([maleiter,femaleiter], shape=(150,257), 
                          mix_method='add', diffseed=True)
FMmixer = SupervisedMixer([femaleiter,maleiter], shape=(150,257), 
                          mix_method='add', diffseed=True)
mixers = [MMmixer, FFmixer, MFmixer, FMmixer]
mixerdesc = ['MM','FF','MF','FM']
mixersSDR = [[],[],[],[]]
mixersSIR = [[],[],[],[]]
mixersSAR = [[],[],[],[]]
    
i=0


num_samples = 500
try:
    starti = i
except:
    starti = 0

for i in range(starti, samples):
    for j,mixer in enumerate(mixers):
        evals = bss_eval_sample(mixer, 2)
        mixersSDR[j].append( 1/(2)*(evals['SDR'][0] + evals['SDR'][1]) )
        mixersSIR[j].append( 1/(2)*(evals['SIR'][0] + evals['SIR'][1]) )
        mixersSAR[j].append( 1/(2)*(evals['SAR'][0] + evals['SAR'][1]) )
        # if j >= 2 and i > 250:
        #     continue
        sys.stdout.write('\r'+str(mixersSDR))


# In[ ]:

MMSDR = np.mean(mixersSDR[0])
FFSDR = np.mean(mixersSDR[1])
MFSDR = np.mean(mixersSDR[2][251:500])
FMSDR = np.mean(mixersSDR[3][251:500])

print('MM: ', MMSDR, ', FF: ', FFSDR, ', MF: ', (MFSDR+FMSDR)/2, ', All: ', (MMSDR+FMSDR+MFSDR+FFSDR)/4)
np.savez( 'In-Set Mixer Metrics.npz', mixersSDR=mixersSDR, mixersSAR=mixersSAR, mixersSIR=mixersSIR)


# ## Out of Set Mixer
omaleiter = SplitsIterator([0.8,0.1,0.1], libritest, speaker_keys=out_set_M, shape=(150,257), return_key=True)
omaleiter.set_split(2)

ofemaleiter = SplitsIterator([0.8,0.1,0.1], libritest, speaker_keys=out_set_F, shape=(150,257), return_key=True)
ofemaleiter.set_split(2)

samples = 500
oMMmixer = SupervisedMixer([omaleiter,omaleiter], shape=(150,257), 
                          mix_method='add', diffseed=True)
oFFmixer = SupervisedMixer([ofemaleiter,ofemaleiter], shape=(150,257), 
                          mix_method='add', diffseed=True)
oMFmixer = SupervisedMixer([omaleiter,ofemaleiter], shape=(150,257), 
                          mix_method='add', diffseed=True)
oFMmixer = SupervisedMixer([ofemaleiter,omaleiter], shape=(150,257), 
                          mix_method='add', diffseed=True)
omixers = [oMMmixer, oFFmixer, oMFmixer, oFMmixer]
omixerdesc = ['MM','FF','MF','FM']
omixersSDR = [[],[],[],[]]
omixersSIR = [[],[],[],[]]
omixersSAR = [[],[],[],[]]
    
i=0


# In[ ]:

num_samples = 500
try:
    starti = i
except:
    starti = 0

for i in range(starti, samples):
    for j,mixer in enumerate(omixers):
        evals = bss_eval_sample(mixer, 2)
        omixersSDR[j].append( 1/(2)*(evals['SDR'][0] + evals['SDR'][1]) )
        omixersSIR[j].append( 1/(2)*(evals['SIR'][0] + evals['SIR'][1]) )
        omixersSAR[j].append( 1/(2)*(evals['SAR'][0] + evals['SAR'][1]) )
        
        sys.stdout.write('\r[i]= '+str(i)+', MM: '+str(np.mean(omixersSDR[0]))+', FF: '+str(FFSDR)+', MF: '+ 
                         str((MFSDR+FMSDR)/2)+', All: '+str((MMSDR+FMSDR+MFSDR+FFSDR)/4))


oMMSDR = np.mean(omixersSDR[0])
oFFSDR = np.mean(omixersSDR[1])
oMFSDR = np.mean(omixersSDR[2])
oFMSDR = np.mean(omixersSDR[3])

print('MM: ', oMMSDR, ', FF: ', oFFSDR, ', MF: ', (oMFSDR+oFMSDR)/2, ', All: ', (oMMSDR+oFMSDR+oMFSDR+oFFSDR)/4)
np.savez( 'Out-of-Set Mixer Metrics.npz', omixersSDR=omixersSDR, omixersSAR=omixersSAR, omixersSIR=omixersSIR)


# # Three Speakers
# 
# ## In-Set Three Speaker Evaluation
fulliter = SplitsIterator([0.8,0.1,0.1], libritrain, shape=(150,257), return_key=True)
fulliter.set_split(2)

samples = 500
mixer = SupervisedMixer([fulliter,fulliter,fulliter], shape=(150,257), 
                          mix_method='add', diffseed=True)
fmixersSDR = []
fmixersSIR = []
fmixersSAR = []
    
i=0

num_samples = 500
try:
    starti = i
except:
    starti = 0

for i in range(starti, samples):
    evals = bss_eval_sample(mixer, 3)
    fmixersSDR.append( 1/(2)*(evals['SDR'][0] + evals['SDR'][1]) )
    fmixersSIR.append( 1/(2)*(evals['SIR'][0] + evals['SIR'][1]) )
    fmixersSAR.append( 1/(2)*(evals['SAR'][0] + evals['SAR'][1]) )
    sys.stdout.write('\r[i]= '+str(i))

np.savez('Three-Speaker In-Set Metrics.npz', fmixersSDR=fmixersSDR, fmixersSIR=fmixersSIR, fmixersSAR=fmixersSAR)


# ## Out-of-Set Three Speaker Evaluation
omaleiter = SplitsIterator([0.8,0.1,0.1], libritest, speaker_keys=out_set_M, shape=(150,257), return_key=True)
omaleiter.set_split(2)
samples = 500

omixer = SupervisedMixer([omaleiter,omaleiter,omaleiter], shape=(150,257), 
                          mix_method='add', diffseed=True)
ofmixersSDR = []
ofmixersSIR = []
ofmixersSAR = []
    
i=0
num_samples = 500
try:
    starti = i
except:
    starti = 0

for i in range(starti, samples):
    evals = bss_eval_sample(omixer, 3)
    ofmixersSDR.append( 1/(2)*(evals['SDR'][0] + evals['SDR'][1]) )
    ofmixersSIR.append( 1/(2)*(evals['SIR'][0] + evals['SIR'][1]) )
    ofmixersSAR.append( 1/(2)*(evals['SAR'][0] + evals['SAR'][1]) )
        
    sys.stdout.write('\r[i]= '+str(i))


ofSDR = np.mean(ofmixersSDR)
print('SDR: ', ofSDR)

np.savez('Three-Speaker Out-Set Metrics.npz', ofmixersSAR=ofmixersSAR, ofmixersSDR=ofmixersSDR,ofmixersSIR=ofmixersSIR)

