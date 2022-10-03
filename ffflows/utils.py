import torch.nn as nn

def getActivation(activation, *args, **kwargs):
    actdict = {
        "linear" : nn.Identity(*args, **kwargs),
        "relu" : nn.ReLU(*args, **kwargs),
        "leakyrelu" : nn.LeakyReLU(*args, **kwargs),
        "sigmoid" : nn.Sigmoid(*args, **kwargs),
        "selu" : nn.SELU(*args, **kwargs),
        "softplus" : nn.Softplus(*args, **kwargs),
    }
    assert lower(activation) in actdict, f"Currently {activation} is not supported"
    
    return actdict(activation, *args, **kwargs)