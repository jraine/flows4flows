import abc
from re import X
from nflows import transforms, flows
from torch.nn import functional as F
import torch.nn as nn
import torch

from ffflows.distance_penalties import BasePenalty


class FlowForFlow(abc.ABC, flows.Flow):
    '''
    Driving class for a flow for flow model.
    Holds the top flow as well as base distributions, and handles training steps for forward and backward.
    '''

    def __init__(self, transform, distribution_right, distribution_left=None, embedding_net=None):
        """Constructor.
        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution_right: A `Distribution` object, the base distribution for the data distribution on the forward pass of the flow
            distribution_left: (Optional) A `Distribution` object, the base distribution for the data distribution on the inverse pass of the flow. If not specified, same as distribution_fwd
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        super().__init__(transform, distribution_right, embedding_net)
        self.base_flow_right = distribution_right
        self._context_used_in_base = True
        self.base_flow_left = distribution_left if distribution_left is not None else distribution_right

        # Distance penalty
        self.distance_object = BasePenalty()

    def add_penalty(self, penalty_object):
        assert isinstance(penalty_object, BasePenalty)
        self.distance_object = penalty_object

    @abc.abstractmethod
    def context_func(self, context_left, context_right):
        """Given the context of the distribution on the left and right return the condition to be used by the flow."""
        return None

    @abc.abstractmethod
    def _direction_func(self, input_context, target_context):
        """Given the context of the input data and the target condition decide if this is going from left to right
        (forward) or right to left (inverse)."""
        return None

    def direction_func(self, input_context, target_context=None):
        if input_context is None:
            return None
        else:
            return self._direction_func(input_context, target_context).view(-1)

    def set_forward_base(self):
        '''Just in case we need to change the base distribution in the subclass'''
        self._distibution = self.base_flow_right

    def set_backward_base(self):
        '''Just in case we need to change the base distribution in the subclass'''
        self._distibution = self.base_flow_left

    def __transform(self, inputs, input_context=None, target_context=None, inverse=False):
        '''Transform inputs with transformer given context.
        Inputs:
            inputs: Input Tensor for transformer
            input_context: Context tensor for samples from left of transformer
            target_context: Context tensor for samples from right of transformer. If None and left is set, uses left
            inverse: In absense of context tensors, specifies if forward or inverse pass of transformer, and thus left
            or right base density. Default False (forward) Choose forward (defualt) or inverse (set to true) pass.
        '''
        if input_context is None:
            context = None
        else:
            if target_context is not None:
                target_context = torch.tile(
                    target_context.view(-1, *input_context.view(input_context.shape[0], -1).shape[1:]),
                    (input_context.shape[0],)).view(input_context.shape) if \
                    target_context.view(-1, *input_context.view(input_context.shape[0], -1).shape[1:]).shape[
                        0] == 1 else target_context
            # The ordering of the contexts needs to remain consistent to ensure invertibility
            if inverse:
                tup = (target_context, input_context)
            else:
                tup = (input_context, target_context)
            context = self._embedding_net(self.context_func(*tup))

        transform = self._transform.inverse if inverse else self._transform
        y, logabsdet = transform(inputs, context=context)

        return y, logabsdet

    def transform(self, inputs, input_context=None, target_context=None, inverse=False):
        '''Transform inputs with transformer given context.
        Inputs:
            inputs: Input Tensor for transformer
            input_context: Context tensor for samples from left of transformer
            target_context: Context tensor for samples from right of transformer. If None and left is set, uses left
            inverse: In absense of context tensors, specifies if forward or inverse pass of transformer, and thus left
            or right base density. Default False (forward) Choose forward (defualt) or inverse (set to true) pass.
        '''
        order = self.direction_func(input_context, target_context)
        if order is None:
            outputs, logabsdet = self.__transform(inputs, input_context, target_context, inverse=inverse)
        else:
            outputs = torch.zeros_like(inputs)
            logabsdet = torch.zeros(len(inputs)).to(inputs)
            for inverse, mx in zip([True, False], [order, ~order]):
                if torch.any(mx):
                    outputs[mx], logabsdet[mx] = self.__transform(inputs[mx], input_context[mx], target_context[mx],
                                                                  inverse=inverse)
        return outputs, logabsdet

    def bd_log_prob(self, noise, input_context=None, target_context=None, inverse=False):
        '''
        Base density log probabilites.
        Inputs:
            noise: Input Tensor for base density
            input_context: Context tensor for samples from left of transformer
            target_context: Context tensor for samples from right of transformer. If None and left is set, uses left
            inverse: In absense of context tensors, specifies if forward or inverse pass of transformer, and thus left
            or right base density. Default False (forward)
        '''
        order = self.direction_func(input_context, target_context)
        if order is None:
            base_flow = self.base_flow_right if inverse is False else self.base_flow_left
            log_prob = base_flow.log_prob(noise)
        else:
            log_prob = torch.zeros(len(noise)).to(noise)
            for base_flow, mx in zip([self.base_flow_left, self.base_flow_right], [order, ~order]):
                if torch.any(mx):
                    log_prob[mx] = base_flow.log_prob(noise[mx], context=target_context[mx])
        return log_prob

    def log_prob(self, inputs, input_context=None, target_context=None, inverse=False):
        '''
        log probability of transformed inputs given context, use relevant base distribution based on forward or inverse, infered from context or specified from inverse
        Inputs:
            inputs: Input Tensor for transformer
            input_context: Context tensor for samples from left of transformer
            target_context: Context tensor for samples from right of transformer. If None and left is set, uses left
            inverse: In absense of context tensors, specifies if forward or inverse pass of transformer, and thus left
            or right base density. Default False (forward) Choose forward (defualt) or inverse (set to true) pass.
        '''
        noise, logabsdet = self.transform(inputs, input_context, target_context, inverse)
        log_prob = self.bd_log_prob(noise, input_context, target_context, inverse)
        dist_pen = -self.distance_object(noise, inputs)
        return log_prob + logabsdet + dist_pen

    def _sample(self, num_samples, context):  # , inverse=False):
        '''
        Something we will never likely need. Sample from a base distribution and pass through the top transformer.
        '''
        raise NotImplementedError()

    def sample_and_log_prob(self, num_samples, context=None):
        raise NotImplementedError()


class NoContextFlowForFlow(FlowForFlow):

    def context_func(self, x, y):
        return torch.zeros_like(x)

    def _direction_func(self, x, y):
        return torch.zeros_like(x, dtype=torch.bool)


class DeltaFlowForFlow(FlowForFlow):

    def context_func(self, x, y):
        return y - x

    def _direction_func(self, x, y):
        return self.context_func(x, y) < 0


class ConcatFlowForFlow(FlowForFlow):

    def context_func(self, x, y):
        return torch.cat([x, y], axis=-1)

    def _direction_func(self, x, y):
        return y - x < 0


class DiscreteBaseFlowForFlow(FlowForFlow):

    def context_func(self, x, y=None):
        return None

    def _direction_func(self, x, y):
        return None


class DiscreteBaseConditionFlowForFlow(FlowForFlow):

    def context_func(self, x, y=None):
        return x

    def _direction_func(self, x, y):
        return None


class BaseFlow(flows.Flow):
    '''
    Wrapper class around Base Flow for a flow for flow model.
    Harmonises function calls with FlowForFlow model.
    Constructed and used exactly like an nflows.Flow object.
    '''

    def log_prob(self, inputs, context=None, input_context=None, target_context=None, inverse=False):
        context = input_context if input_context is not None else context
        return super(BaseFlow, self).log_prob(inputs, context=context)
