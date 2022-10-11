import torch.nn as nn
from torch.nn import functional as F

def set_trainable(model,trainable=True):
    for param in model.parameters():
        param.requires_grad = trainable
    model.eval()

# def get_activation(activation, *args, **kwargs):
#     actdict = {
#         "linear" : nn.Identity,
#         "relu" : nn.ReLU,
#         "leakyrelu" : nn.LeakyReLU,
#         "sigmoid" : nn.Sigmoid,
#         "selu" : nn.SELU,
#         "softplus" : nn.Softplus,
#     }
#     assert activation.lower() in actdict, f"Currently {activation} is not supported"
    
#     return actdict(activation.lower())(*args, **kwargs)
