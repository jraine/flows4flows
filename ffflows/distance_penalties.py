from copy import deepcopy

import numpy as np
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


class AnnealedPenalty(BasePenalty):

    def __init__(self, penalty, n_steps=None, min_weight=0):
        super(AnnealedPenalty, self).__init__()
        self.min_weight = min_weight
        self.penalty_to_wrap = penalty
        self.initial_weight = deepcopy(self.penalty_to_wrap.weight)
        self.n_steps = n_steps
        self.pi = torch.tensor(np.pi, dtype=torch.float32)
        self.step = 0

    def set_n_steps(self, n_steps):
        if self.n_steps is None:
            self.n_steps = n_steps

    def update_weight(self):
        # TODO currently following a cosine schedule, but this should be possible to set/configure.
        self.penalty_to_wrap.weight = self.min_weight + 0.5 * (self.initial_weight - self.min_weight) * (
                    1 + (self.pi * self.step / self.n_steps).cos())
        self.step += 1

    def penalty_function(self, inputs, outputs):
        self.update_weight()
        return self.penalty_to_wrap.penalty_function(inputs, outputs)
