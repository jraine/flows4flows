from nflows import transforms
from torch.nn import functional as F

from .utils import getActivation


class DenseNet(nn.Module):
    '''Simple feedforward network'''
    def __init__(self, input_dim, output_dim, hidden_nodes, hidden_act='relu', output_act='linear',
                 batch_norm=False, layer_norm=False, dropout=0, bias=True, *args, **kwargs):
        '''Constructor, provide the input and output dimensions.
        List of hidden nodes.
        Choose activations as strings.
        '''
        super(self,DenseNet).__init__(*args,**kwargs)
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

class FlowForFlowModel(nn.module):
    '''Driving class for a flow for flow model.
    Holds the top flow as well as base distributions, and handles training steps for forward and backward.
    '''
    def __init__(self):
        super(self,FlowForFlowModel).__init__()
        self.transform = None
        self.base_flow_fwd = None
        self.base_flow_inv = None
        self.configured = 0b000

    def setTransform(self, transform):
        '''Set the transformer for the flows4flows model.
        This should be an nflows transformer'''
        self.transform = transform
        self.configured |= 1

    def setBaseFlows(self, base_fwd, base_inv=None):
        '''Set the base distribution flows for the forward (and optionally inverse if different) passes.'''
        self.base_flow_fwd = base_fwd
        self.configured |= 1<<1
        if base_inv is not None:
            self.base_flow_inv = base_inv
        else:
            self.base_flow_inv = base_fwd
        self.configured |= 1<<2
    
    def transform_and_log_prob(self, x, context=None, inverse=False):
        '''Transform inputs through top transformer. Inverse pass possible.
        Optionally pass a context tuple. Each element of the tuple will be passed as the context to the respective base distribution.'''
        assert self.configured & 0b111, "Must have top flow and base flows configured"
        if context is None:
            context_l,context_r = (None, None)
        else:
            context_l,context_r = context
        
        y, logabsdet = self.forward(x,context,inverse)
        if inverse:
            logprob = self.base_flow_inv._log_prob(y, context_l)
        else:
            logprob = self.base_flow_fwd._log_prob(y, context_r)
        return y, logprob - logabsdet
    
    def forward(self, x, context=None, inverse=False):
        assert self.configured & 0b001, "Must have top flow configured"
        # context_l,context_r = context
        if inverse:
            y, logabsdet = self.topflow._transform.inverse(x, context=context)
        else:
            y, logabsdet = self.topflow._transform(x, context=context)
        
        return y, logabsdet

    def _log_prob(self, x, context=None, inverse=False):
        _, logprob = self.transform_and_log_prob(x, context=context, inverse=inverse)
        return logprob


    def sample(self, num_samples, context, inverse=False):
        '''Something we will never likely need. Sample from a base distribution and pass through the top transformer.'''
        assert self.configured & 0b111, "Must have top flow and base flows configured"
        if context is None:
            context_l,context_r = (None, None)
        else:
            context_l,context_r = context
        if inverse:
            x = self.base_flow_fwd._sample(num_samples, context = context_r)
            samples = self.transform.inverse(x, context = context)
        else:
            x = self.base_flow_inv._sample(num_samples, context = context_l)
            samples = self.transform(x, context = context)
        return samples

    def forward(self, x, context, inverse=False):
        y, log_prob = self._log_prob(x, context=context,inverse=inverse)