import sys
from . import app

#sys.path.append(app.root_path + '/../..')



#from .cnn_models import Conv1DModel
#from .model_functions import separate_sources

import soundfile as sf
import numpy as np
from magnolia.dnnseparate.L41model import L41Model
from magnolia.dnnseparate.deep_clustering_model import DeepClusteringModel
#from .deep_clustering_models import DeepClusteringModel
#from .clustering_utils import clustering_separate
from magnolia.utils.clustering_utils import clustering_separate,preprocess_signal
#from .l41_models import L41Model
from magnolia.factorization.nmf import easy_nmf_separate
from magnolia.features.preprocessing import undo_preemphasis
from magnolia.features.spectral_features import istft
'''
Input: noisy signal path
Output: list of separated speakers (in numpy form)
'''

'''def tflow_separate(input_path):
    
    rate = 10000

    wav_list = []

    model = Conv1DModel([None,None,251,1],[None,None,251,2],20,600,50)
    model.load(app.root_path + '/static/models/better_cnn-mask-model.ckpt')

    outputs = separate_sources(input_path,model)
    for row in outputs:
        wav_list.append(row)

    return wav_list 
'''
'''
Input: noisy signal path
Output: list of separated speakers (in numpy form)
'''
def deep_cluster_separate(input_path):

    
    wav_list = []

    input_signal,sample_rate = sf.read(input_path)

    model  = DeepClusteringModel()
    model.load(app.root_path + '/static/models/deep_clustering.ckpt')

    outputs = clustering_separate(input_signal,sample_rate,model,2)
    for row in outputs:
        wav_list.append(row)

    return wav_list 


'''
Input: noisy signal path
Output: list of separated speakers (in numpy form)
'''
def l41_separate(input_path):
    wav_list = []

    input_signal,sample_rate = sf.read(input_path)

    model  = L41Model(nonlinearity='tanh', normalize=False)
    model.load(app.root_path + '/static/models/lab41_nonorm-final.ckpt')

    outputs = clustering_separate(input_signal,sample_rate,model,2)
    for row in outputs:
        wav_list.append(row)

    return wav_list 

def nmf_sep(input_path):

    wav_list = []

    input_signal,sample_rate = sf.read(input_path)

    # Preprocess the signal into an input feature
    spectrogram, X_in = preprocess_signal(input_signal, sample_rate)

    separated_speakers = np.square(easy_nmf_separate(spectrogram))
    phases = np.unwrap(np.angle(spectrogram))

    # Invert the STFT to recover the output waveforms, remembering to undo the
    # preemphasis
    waveforms = []
    for i in range(2):
        waveform = istft(separated_speakers[i]*np.exp(phases[i]*1.0j), 1e4, None, 0.0256, two_sided=False,fft_size=512)
        unemphasized = undo_preemphasis(waveform)
        waveforms.append(unemphasized)

    sources = np.stack(waveforms)

    for row in sources:
        wav_list.append(row)

    return wav_list 


