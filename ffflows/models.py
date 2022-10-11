import abc
from re import X
from nflows import transforms, flows
from torch.nn import functional as F
import torch.nn as nn
import torch

# class DenseNet(nn.Module):
#     '''Simple feedforward network'''

#     def __init__(self, input_dim, output_dim, hidden_nodes, hidden_act='relu', output_act='linear',
#                  batch_norm=False, layer_norm=False, dropout=0, bias=True, *args, **kwargs):
#         '''Constructor, provide the input and output dimensions.
#         List of hidden nodes.
#         Choose activations as strings.
#         '''
#         super().__init__(*args, **kwargs)
#         self.layers = []
#         for f_in, f_out in zip([input_dim] + hidden_nodes[:-1], hidden_nodes):
#             self.layers.append(nn.linear(f_in, f_out, bias=bias))
#             self.layers.append(nn.linear(f_in, f_out, bias=bias))
#             self.layers.append(get_activation(hidden_act))
#             if batch_norm:
#                 self.layers.append(nn.BatchNorm1D(f_out))
#             if layer_norm:
#                 self.layers.append(nn.LayerNorm(f_out))
#             if dropout != 0:
#                 assert type(dropout) is float, "Dropout value must be a float between 0 and 1"
#                 assert (dropout >= 0) & (dropout < 1), "Dropout value must be a float between 0 and 1"
#                 self.layers.append(nn.Dropout(dropout))
#         self.layers.append(nn.linear(hidden_nodes[-1], output_dim, bias=bias))
#         self.layers.append(get_activation(output_act))

#     def forward(self, input):
#         x = input
#         for layer in self.layers:
#             x = layer(x)
#         return x


# class FlowForFlow(flows.Flow):
#     '''
#     Driving class for a flow for flow model.
#     Holds the top flow as well as base distributions, and handles training steps for forward and backward.
#     '''
#
#     def __init__(self, transform, distribution_fwd, distribution_inv=None, embedding_net=None, context_func=None):
#         super().__init__(transform, distribution_fwd, embedding_net)
#         self.base_flow_fwd = distribution_fwd
#         self._context_used_in_base = True
#         self.base_flow_inv = distribution_inv if distribution_inv is not None else distribution_fwd
#         self.context_func = context_func if context_func is not None else lambda x, y: torch.cat((x, y), axis=-1)
#
#     def set_forward_base(self):
#         '''Just in case we need to change the base distribution in the subclass'''
#         self._distibution = self.base_flow_fwd
#
#     def set_backward_base(self):
#         '''Just in case we need to change the base distribution in the subclass'''
#         self._distibution = self.base_flow_inv
#
#     def transform(self, inputs, context=None, inverse=False):
#         '''Transform inputs with transformer given context. Choose dorward (defualt) or inverse (set to true) pass'''
#         if context is not None:
#             context_l, context_r = context
#             context = self._embedding_net(self.context_func(context_r, context_l))
#
#         if inverse:
#             y, logabsdet = self._transform.inverse(inputs, context=context)
#         else:
#             y, logabsdet = self._transform(inputs, context=context)
#
#         return y, logabsdet
#
#     def log_prob(self, inputs, context=None, inverse=False):
#         '''
#         log probability of transformed inputs given context, use relevant base distribution based on forward or inverse.
#         '''
#         if context is None:
#             context_l, context_r = (None, None)
#         else:
#             context_l, context_r = context
#         # embedded_context = self._embedding_net(self.context_func(context_r,context_l))
#
#         noise, logabsdet = self.transform(inputs, context=context, inverse=inverse)
#         if context is not None:
#             con = context_l if inverse else context_r
#             log_prob = self._distribution.log_prob(noise, context=con)
#         else:
#             log_prob = self._distribution.log_prob(noise)
#         return log_prob + logabsdet
#
#     def _sample(self, num_samples, context):  # , inverse=False):
#         '''
#         Something we will never likely need. Sample from a base distribution and pass through the top transformer.
#         '''
#         raise NotImplementedError()  # "Shouldn't really want to sample from a base distribution with flows for flows"
#         # if context is None:
#         #     context_l,context_r = (None, None)
#         # else:
#         #     embedded_context = self._embedding_net(context)
#         #     context_l,context_r = context
#         # if inverse:
#         #     x = self.base_flow_fwd._sample(num_samples, context = context_r)
#         #     samples,_ = self._passthrough(x, context=embedded_context, inverse=inverse)
#         # else:
#         #     x = self.base_flow_inv._sample(num_samples, context = context_l)
#         #     samples,_ = self._passthrough(x, context=embedded_context, inverse=inverse)
#         # return samples
#
#     # def forward(self, x, context, inverse=False):
#     #     y, log_prob = self._log_prob(x, context=context,inverse=inverse)
#
#     def sample_and_log_prob(self, num_samples, context=None):
#         raise NotImplementedError()


class FlowForFlow(abc.ABC, flows.Flow):
    '''
    Driving class for a flow for flow model.
    Holds the top flow as well as base distributions, and handles training steps for forward and backward.
    '''

    def __init__(self, transform, distribution_fwd, distribution_inv=None, embedding_net=None):
        super().__init__(transform, distribution_fwd, embedding_net)
        self.base_flow_fwd = distribution_fwd
        self._context_used_in_base = True
        self.base_flow_inv = distribution_inv if distribution_inv is not None else distribution_fwd

    @abc.abstractmethod
    def context_func(self, context_l, context_r):
        return None

    @abc.abstractmethod
    def _direction_func(self, context_l, context_r):
        return None

    def direction_func(self, context_l, context_r=None):
        if context_l is None:
            return None
        else:
            return self._direction_func(context_l, context_r).view(-1)

    def set_forward_base(self):
        '''Just in case we need to change the base distribution in the subclass'''
        self._distibution = self.base_flow_fwd

    def set_backward_base(self):
        '''Just in case we need to change the base distribution in the subclass'''
        self._distibution = self.base_flow_inv

    def __transform(self, inputs, context_l=None, context_r=None, inverse=False):
        '''Transform inputs with transformer given context.
        Inputs:
            inputs: Input Tensor for transformer
            context_l: Context tensor for samples from left of transformer
            context_r: Context tensor for samples from right of transformer. If None and left is set, uses left
            inverse: In absense of context tensors, specifies if forward or inverse pass of transformer, and thus left \
or right base density. Default False (forward) Choose forward (defualt) or inverse (set to true) pass.'''
        if context_l is None:
            context = None
        else:
            if context_r is not None:
                context_r = torch.tile(context_r.view(-1,*context_l.view(context_l.shape[0],-1).shape[1:]),(context_l.shape[0],)).view(context_l.shape) if context_r.view(-1,*context_l.view(context_l.shape[0],-1).shape[1:]).shape[0] == 1 else context_r
            context = self._embedding_net(self.context_func(context_r, context_l))

        transform = self._transform.inverse if inverse else self._transform
        y, logabsdet = transform(inputs, context=context)

        return y, logabsdet

    def transform(self, inputs, context_l=None, context_r=None, inverse=False):
        '''Transform inputs with transformer given context.
        Inputs:
            inputs: Input Tensor for transformer
            context_l: Context tensor for samples from left of transformer
            context_r: Context tensor for samples from right of transformer. If None and left is set, uses left
            inverse: In absense of context tensors, specifies if forward or inverse pass of transformer, and thus left \
or right base density. Default False (forward) Choose forward (defualt) or inverse (set to true) pass.'''
        order = self.direction_func(context_l, context_r)
        if order is None:
            outputs, logabsdet = self.__transform(inputs, context_l, context_r, inverse=inverse)
        else:
            outputs = torch.zeros_like(inputs)
            logabsdet = torch.zeros(len(inputs)).to(inputs)
            for direction, mx in zip([True, False], [order, ~order]):
                outputs[mx], logabsdet[mx] = self.__transform(inputs[mx], context_l[mx], context_r[mx],
                                                              inverse=direction)
        return outputs, logabsdet

    def bd_log_prob(self, noise, context_l=None, context_r=None, inverse=False):
        '''
        Base density log probabilites.
        Inputs:
            noise: Input Tensor for base density
            context_l: Context tensor for samples from left of transformer
            context_r: Context tensor for samples from right of transformer. If None and left is set, uses left
            inverse: In absense of context tensors, specifies if forward or inverse pass of transformer, and thus left or right base density. Default False (forward)
        '''
        order = self.direction_func(context_l, context_r)
        if order is None:
            base_flow = self.base_flow_fwd if inverse is False else self.base_flow_inv
            log_prob = base_flow.log_prob(noise)
        else:
            log_prob = torch.zeros(len(noise)).to(noise)
            log_prob[order] = self.base_flow_fwd.log_prob(noise[order], context=context_r[order])
            log_prob[~order] = self.base_flow_inv.log_prob(noise[~order], context=context_l[~order])
        return log_prob

    def log_prob(self, inputs, context_l=None, context_r=None, inverse=False):
        '''
        log probability of transformed inputs given context, use relevant base distribution based on forward or inverse, infered from context or specified from inverse
        Inputs:
            inputs: Input Tensor for transformer
            context_l: Context tensor for samples from left of transformer
            context_r: Context tensor for samples from right of transformer. If None and left is set, uses left
            inverse: In absense of context tensors, specifies if forward or inverse pass of transformer, and thus left \
or right base density. Default False (forward) Choose forward (defualt) or inverse (set to true) pass.
        '''

        noise, logabsdet = self.transform(inputs, context_l, context_r, inverse)
        log_prob = self.bd_log_prob(noise, context_l, context_r, inverse)
        return log_prob + logabsdet

    def _sample(self, num_samples, context):  # , inverse=False):
        '''
        Something we will never likely need. Sample from a base distribution and pass through the top transformer.
        '''
        raise NotImplementedError()

    def sample_and_log_prob(self, num_samples, context=None):
        raise NotImplementedError()


class DeltaFlowForFlow(FlowForFlow):

    def context_func(self, x, y):
        return y - x

    def _direction_func(self, x, y):
        return self.context_func(x, y) < 0

class ConcatFlowForFlow(FlowForFlow):

    def context_func(self, x, y):
        return torch.cat([x,y],axis=-1)

    def _direction_func(self, x, y):
        return self.context_func(x, y) < 0

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