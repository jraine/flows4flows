import pathlib

import torch
from torch.nn import functional as F

from ffflows.models import DeltaFlowForFlow, ConcatFlowForFlow, DiscreteBaseFlowForFlow, DiscreteBaseConditionFlowForFlow
from ffflows.data.plane import ConditionalAnulus, Anulus, ConcentricRings, FourCircles, CheckerboardDataset

from nflows import transforms
from nflows.distributions import StandardNormal
from nflows.flows import Flow



def get_data(name, *args, **kwargs):
    datadict = {
        "conditionalanulus" : ConditionalAnulus,
        "anulus" : Anulus,
        "concentricrings" : ConcentricRings,
        "fourcircles" : FourCircles,
        "checkerboard" : CheckerboardDataset
    }
    assert name.lower() in datadict.keys(), f"Currently {name} is not supported. Choose one of {datadict.keys()}"

    return datadict[name.lower()](*args, **kwargs)

def get_flow4flow(name, *args, **kwargs):
    f4fdict = {
        "delta" : DeltaFlowForFlow,
        "concat" : ConcatFlowForFlow,
        "discretebase" : DiscreteBaseFlowForFlow,
        "discretebasecondition" : DiscreteBaseConditionFlowForFlow,
    }
    assert name.lower() in f4fdict, f"Currently {f4fdict} is not supported"

    return f4fdict[name](*args, **kwargs)


def spline_inn(inp_dim, nodes=128, num_blocks=2, num_stack=3, tail_bound=3.5, tails='linear', activation=F.relu, lu=0,
               num_bins=10, context_features=None):
    transform_list = []
    for i in range(num_stack):
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


def train(model, train_data, val_data, n_epochs, learning_rate, ncond, path, name, rand_perm_target, loss_fig=True, device='cpu'):
    
    save_path = pathlib.Path(path / name)
    save_path.mkdir(parents=True,exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_steps = len(train_data) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, last_epoch=-1,
                                                           eta_min=0)
    train_loss = torch.zeros(n_epochs)
    valid_loss = torch.zeros(n_epochs)
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        t_loss = []

        for step, data in enumerate(train_data):
            model.train()

            optimizer.zero_grad()
            if ncond is not None:
                inputs, context_l = data
                context_r = torch.rand_perm(context_l) if rand_perm_target else None
            else:
                inputs, context_l, context_r = data, None, None
            
            _, logprob = -model.transform(inputs, context_l=context_l, context_r=context_r)

            logprob.backward()
            optimizer.step()
            scheduler.step()
            t_loss.append(logprob.item())
        
        train_loss[epoch] = torch.tensor(t_loss).mean()
        v_loss = torch.zeros(len(val_data))
        for v_step, data in enumerate(val_data):
            if ncond is not None:
                inputs, context_l = data
                context_r = torch.rand_perm(context_l) if rand_perm_target else None
            else:
                inputs, context_l, context_r = data, None, None

            with torch.no_grad():
                v_loss[v_step] = -model.log_prob(inputs, context_l=context_l, context_r=context_r)
        valid_loss[epoch] = v_loss.mean()

        torch.save(model.state_dict(), save_path / f'epoch_{epoch}_valloss_{valid_loss[epoch]}')
        print(f"Loss = {train_loss[epoch]:.3f},\t val_loss = {val_loss[epoch]:.3f}")

    ###insert saving of losses and plots and stuff
    if loss_fig:
        # Training and validation losses
        fig = plot_training(train_loss, valid_loss)
        fig.savefig(sv_nm)
        # fig.show()
        plt.close(fig)

    model.eval()        
    return train_loss, valid_loss