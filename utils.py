from .models import DeltaFlowForFlow, ConcatFlowForFlow, DiscreteBaseFlowForFlow, DiscreteBaseConditionFlowForFlow
from .data.plane import ConditionalAnulus, Anulus, ConcentricRings, FourCircles, CheckerboardDataset

from nflows import transforms
from nflows.distributions import StandardNormal
from nflows.flows import Flow

def get_data(name, *args, **kwargs):
    datadict = {
        "ConditionalAnulus" : ConditionalAnulus,
        "Anulus" : Anulus,
        "ConcentricRings" : ConcentricRings,
        "FourCircles" : FourCircles,
        "CheckerboardDataset" : CheckerboardDataset
    }
    assert name.lower() in datadict, f"Currently {datadict} is not supported"

    return datadict(name)

def get_flow4flow(name, *args, **kwargs):
    f4fdict = {
        "Delta" : DeltaFlowForFlow,
        "Concat" : ConcatFlowForFlow,
        "DiscreteBase" : DiscreteBaseFlowForFlow,
        "DiscreteBaseCondition" : DiscreteBaseConditionFlowForFlow,
    }
    assert name.lower() in f4fdict, f"Currently {f4fdict} is not supported"

    return f4fdict(name, *args, **kwargs)


def spline_inn(inp_dim, nodes=128, nblocks=2, nstack=3, tail_bound=3.5, tails='linear', activation=F.relu, lu=0,
               num_bins=10, context_features=None):
    transform_list = []
    for i in range(nstack):
        transform_list += [
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, nodes,
                                                                               num_blocks=num_blocks,
                                                                               tail_bound=tail_bound,
                                                                               num_bins=num_bins,
                                                                               tails=tails, activation=activation,
                                                                               context_features=context_features)]
        if lu:
            transform_list += [transforms.LULinear(inp_dim)]
        else:
            transform_list += [transforms.ReversePermutation(inp_dim)]

    return transforms.CompositeTransform(transform_list[:-1])