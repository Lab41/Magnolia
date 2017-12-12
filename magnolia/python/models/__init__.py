from .dnnseparate.chimera import Chimera
from .dnnseparate.L41_regression_model import L41RegressionModel

__all__ = [
    "Chimera",
    "L41RegressionModel",
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
