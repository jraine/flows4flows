from re import X
from nflows import transforms, flows
from torch.nn import functional as F
import torch.nn as nn

from .utils import getActivation


class DenseNet(nn.Module):
    '''Simple feedforward network'''
    def __init__(self, input_dim, output_dim, hidden_nodes, hidden_act='relu', output_act='linear',
                 batch_norm=False, layer_norm=False, dropout=0, bias=True, *args, **kwargs):
        '''Constructor, provide the input and output dimensions.
        List of hidden nodes.
        Choose activations as strings.
        '''
        super().__init__(*args,**kwargs)
        self.layers = []
        for f_in,f_out in zip([input_dim]+hidden_nodes[:-1],hidden_nodes):
            self.layers.append(nn.linear(f_in,f_out,bias=bias))
            self.layers.append(nn.linear(f_in,f_out,bias=bias))
            self.layers.append(getActivation(hidden_act))
            if batch_norm:
                self.layers.append(nn.BatchNorm1D(f_out))
            if layer_norm:
                self.layers.append(nn.LayerNorm(f_out))
            if dropout != 0:
                assert type(dropout) is float, "Dropout value must be a float between 0 and 1"
                assert (dropout >= 0) & (dropout < 1), "Dropout value must be a float between 0 and 1"
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.linear(hidden_nodes[-1],output_dim,bias=bias))
        self.layers.append(getActivation(output_act))
        

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        return x

class FlowForFlow(flows.Flow):
    '''Driving class for a flow for flow model.
    Holds the top flow as well as base distributions, and handles training steps for forward and backward.
    '''
    def __init__(self, transform, distribution_fwd, distribution_inv=None, embedding_net=None, context_func=None):
        super().__init__(transform, distribution_fwd, embedding_net)
        self.base_flow_fwd = distribution_fwd
        self._context_used_in_base = True
        self.base_flow_inv = distribution_inv if distribution_inv is not None else distribution_fwd
        self.context_func = context_func if context_func is not None else nn.Identity
        
    def set_forward_base(self):
        '''Just in case we need to change the base distribution in the subclass'''
        self._distibution = self.base_flow_fwd
        
    def set_backward_base(self):
        '''Just in case we need to change the base distribution in the subclass'''
        self._distibution = self.base_flow_inv
    
    def transform(self,inputs, context=None, inverse=False):
        '''Transform inputs with transformer given context. Choose dorward (defualt) or inverse (set to true) pass'''
        if context is None:
            context_l,context_r = (None, None)
        else:
            context_l,context_r = context
            context = self._embedding_net(self.context_func(context_r,context_l))
            
        if inverse:
            y, logabsdet = self._transform.inverse(inputs, context=context)
        else:
            y, logabsdet = self._transform(inputs, context=context)
        
        return y, logabsdet

    def log_prob(self, inputs, context=None, inverse=False):
        '''log probability of transformed inputs given context, use relevant base distribution based on forward or inverse.'''
        if context is None:
            context_l,context_r = (None, None)
        else:
            context_l,context_r = context
        # embedded_context = self._embedding_net(self.context_func(context_r,context_l))
        
        noise, logabsdet = self.transform(inputs, context=context, inverse=inverse)
        if context is not None:
            con = context_l if inverse else context_r
            log_prob = self._distribution.log_prob(noise, context=con)
        else:
            log_prob = self._distribution.log_prob(noise)
        return log_prob + logabsdet


    def _sample(self, num_samples, context):#, inverse=False):
        '''Something we will never likely need. Sample from a base distribution and pass through the top transformer.'''
        raise NotImplementedError()# "Shouldn't really want to sample from a base distribution with flows for flows"
        # if context is None:
        #     context_l,context_r = (None, None)
        # else:
        #     embedded_context = self._embedding_net(context)
        #     context_l,context_r = context
        # if inverse:
        #     x = self.base_flow_fwd._sample(num_samples, context = context_r)
        #     samples,_ = self._passthrough(x, context=embedded_context, inverse=inverse)
        # else:
        #     x = self.base_flow_inv._sample(num_samples, context = context_l)
        #     samples,_ = self._passthrough(x, context=embedded_context, inverse=inverse)
        # return samples

    # def forward(self, x, context, inverse=False):
    #     y, log_prob = self._log_prob(x, context=context,inverse=inverse)

    def sample_and_log_prob(self, num_samples, context=None):
        raise NotImplementedError()
