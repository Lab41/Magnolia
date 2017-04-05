import sys
from . import app

sys.path.append(app.root_path + '/../..')

from .cnn_models import Conv1DModel
from .model_functions import separate_sources
import soundfile as sf


'''
Input: noisy signal path
Output: list of separated speakers (in numpy form)
'''

def tflow_separate(input_path):
	
	rate = 10000

	wav_list = []

	model = Conv1DModel([None,None,251,1],[None,None,251,2],20,600,50)
	model.load(app.root_path + '/static/models/cnn-mask-model.ckpt')

	outputs = separate_sources(input_path,model)
	for row in outputs:
		wav_list.append(row)

	return wav_list	
