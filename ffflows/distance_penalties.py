import torch
from torch import nn


class BasePenalty(nn.Module):

    def __init__(self, weight):
        super(BasePenalty, self).__init__()
        self.register_buffer('weight', torch.Tensor(weight))

    def forward(self, inputs, outputs):
        return self.weight * 0


class CustomPenalty(BasePenalty):
    def __init__(self, lambda_func, weight=1.0):
        self.lambda_fn = lambda_func
        super(CustomPenalty, self).__init__(weight)

    def forward(self, inputs, outputs):
        return self.weight * self.lambda_fn(inputs, outputs)


class LOnePenalty(BasePenalty):

    def forward(self, inputs, outputs):
        return self.weight * torch.nn.L1Loss(reduction='none')(outputs - inputs)


class LTwoPenalty(BasePenalty):

    def forward(self, inputs, outputs):
        return self.weight * torch.nn.MSELoss(reduction='none')(outputs - inputs)
