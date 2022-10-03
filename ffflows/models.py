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

class JointBaseDist(nn.module):
    '''Envelope class for flow4flow with joint base distribution'''
    def __init__(self,feature_dim, condition_dim, top_flow, base_flow, *args, **kwargs):
        super(self,JointBaseDist).__init__(*args,**kwargs)

class SeparateBaseDist(nn.module):
    '''Envelope class for flow4flow with separate base distributions'''
    def __init__(self, feature_dim, condition_dim, top_flow, base_flow_fwd, base_flow_inv, *args, **kwargs):
        super(self,SeparateBaseDist).__init__(*args,**kwargs)