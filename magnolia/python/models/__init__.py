from .dnnseparate.jflec import JFLEC
from .dnndenoise.chimera import Chimera
from .dnndenoise.L41_regression_model import L41RegressionModel
from .dnndenoise.sce_mask import RatioMaskSCE

__all__ = [
    "JFLEC",
    "Chimera",
    "L41RegressionModel",
    "RatioMaskSCE",
]

def make_model(model_name, config):
    if model_name in __all__:
        return globals()[model_name](config)
    else:
        raise Exception('The model name %s does not exist' % model_name)

def get_model_class(model_name):
    if model_name in __all__:
        return globals()[model_name]
    else:
        raise Exception('The model name %s does not exist' % model_name)
