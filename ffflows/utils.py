import torch.nn as nn
from torch.nn import functional as F

def set_trainable(model,trainable=True):
    for param in model.parameters():
        param.requires_grad = trainable
    model.eval()