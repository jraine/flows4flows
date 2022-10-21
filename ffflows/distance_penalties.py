import torch
from torch import nn


class BasePenalty(nn.Module):

    def __init__(self, weight=0.0):
        super(BasePenalty, self).__init__()
        self.register_buffer('weight', torch.Tensor([weight]))

    def penalty_function(self, inputs, outputs):
        return 0

    def forward(self, inputs, outputs):
        return self.weight * self.penalty_function(inputs, outputs)


class CustomPenalty(BasePenalty):
    def __init__(self, lambda_func, weight=1.0):
        self.lambda_fn = lambda_func
        super(CustomPenalty, self).__init__(weight)

    def penalty_function(self, inputs, outputs):
        return self.lambda_fn(inputs, outputs)


class WrapPytorchPenalty(BasePenalty):

    def __init__(self, pytorch_method, weight):
        super(WrapPytorchPenalty, self).__init__(weight)
        self.torch_penalty = pytorch_method(reduction='none')

    def penalty_function(self, inputs, outputs):
        return self.torch_penalty(outputs, inputs).sum(-1)


class LOnePenalty(WrapPytorchPenalty):

    def __init__(self, weight):
        super(LOnePenalty, self).__init__(nn.L1Loss, weight)


class LTwoPenalty(WrapPytorchPenalty):

    def __init__(self, weight):
        super(LTwoPenalty, self).__init__(nn.MSELoss, weight)
