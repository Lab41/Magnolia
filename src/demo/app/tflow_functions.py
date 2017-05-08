import sys
from . import app

sys.path.append(app.root_path + '/../..')

from .cnn_models import Conv1DModel
from .model_functions import separate_sources
import soundfile as sf
from .deep_clustering_models import DeepClusteringModel
from .clustering_utils import clustering_separate

'''
Input: noisy signal path
Output: list of separated speakers (in numpy form)
'''

def tflow_separate(input_path):
	
	rate = 10000

	wav_list = []

	model = Conv1DModel([None,None,251,1],[None,None,251,2],20,600,50)
	model.load(app.root_path + '/static/models/better_cnn-mask-model.ckpt')

	outputs = separate_sources(input_path,model)
	for row in outputs:
		wav_list.append(row)

	return wav_list	

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
	print("hello")
