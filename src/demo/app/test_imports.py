import sys

sys.path.append('/Users/aganesh/PublicLab41/Magnolia/src/')

from cnn_models import Conv1DModel
from model_functions import separate_sources

model = Conv1DModel([None,None,251,1],[None,None,251,2],20,600,50)

model.load('/Users/aganesh/PublicLab41/Magnolia/src/demo/app/static/cnn-mask-model.ckpt')

outputs = separate_sources('/Users/aganesh/PublicLab41/Magnolia/src/demo/app/static/mixed_signal.wav',model)

print("Output size", outputs.shape)