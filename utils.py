import pathlib

import torch
from torch.nn import functional as F

from ffflows.models import DeltaFlowForFlow, ConcatFlowForFlow, DiscreteBaseFlowForFlow, DiscreteBaseConditionFlowForFlow
from ffflows.data.plane import ConditionalAnulus, Anulus, ConcentricRings, FourCircles, CheckerboardDataset, TwoSpiralsDataset, FixedWidthAnulus
from ffflows.utils import shuffle_tensor

from nflows import transforms
from nflows.distributions import StandardNormal
from nflows.flows import Flow

from plot import plot_training
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

def get_activation(name, *args, **kwargs):
    actdict = {
        "linear" : lambda x: x,
        "relu" : F.relu,
        "leaky_relu" : F.leaky_relu,
        "sigmoid" : F.sigmoid,
        "selu" : F.selu,
        "celu" : F.celu,
        "elu" : F.elu,
        "swish" : F.hardswish,
        "softplus" : F.softplus,
    }
    assert name.lower() in actdict, f"Currently {name} is not supported.  Choose one of '{actdict.keys()}'"

    return actdict[name.lower()]

def get_data(name, num_points, *args, **kwargs):
    datadict = {
        "conditionalanulus" : ConditionalAnulus,
        "anulus" : Anulus,
        "ring": FixedWidthAnulus,
        "concentricrings" : ConcentricRings,
        "fourcircles" : FourCircles,
        "checkerboard" : CheckerboardDataset,
        "spirals" : TwoSpiralsDataset,
    }
    assert name.lower() in datadict.keys(), f"Currently {name} is not supported. Choose one of '{datadict.keys()}'"
    # batch_size = num_points if batch_size is None else batch_size
    if name.lower() == 'ring':
        return datadict[name.lower()](num_points, radius=0.5)
    else:
        return datadict[name.lower()](num_points)
    # return datadict[name.lower()](num_points)

def get_flow4flow(name, *args, **kwargs):
    f4fdict = {
        "delta" : DeltaFlowForFlow,
        "concat" : ConcatFlowForFlow,
        "discretebase" : DiscreteBaseFlowForFlow,
        "discretebasecondition" : DiscreteBaseConditionFlowForFlow,
    }
    assert name.lower() in f4fdict, f"Currently {f4fdict} is not supported. Choose one of '{f4fdict.keys()}'"

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


def train(model, train_data, val_data, n_epochs, learning_rate, ncond, path, name, rand_perm_target=False, inverse=False, loss_fig=True, device='cpu'):
    
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
                context_r = shuffle_tensor(context_l) if rand_perm_target else None
            else:
                inputs, context_l, context_r = data, None, None
            
            logprob = -model.log_prob(inputs, context_l=context_l, context_r=context_r, inverse=inverse).mean()
            
            logprob.backward()
            optimizer.step()
            scheduler.step()
            t_loss.append(logprob.item())
        
        train_loss[epoch] = torch.tensor(t_loss).mean()
        v_loss = torch.zeros(len(val_data))
        for v_step, data in enumerate(val_data):
            if ncond is not None:
                inputs, context_l = data
                context_r = shuffle_tensor(context_l) if rand_perm_target else None
            else:
                inputs, context_l, context_r = data, None, None

            with torch.no_grad():
                v_loss[v_step] = -model.log_prob(inputs, context_l=context_l, context_r=context_r, inverse=inverse).mean()
        valid_loss[epoch] = v_loss.mean()

        torch.save(model.state_dict(), save_path / f'epoch_{epoch}_valloss_{valid_loss[epoch]:.3f}.pt')
        print(f"Loss = {train_loss[epoch]:.3f},\t val_loss = {valid_loss[epoch]:.3f}")

    ###insert saving of losses and plots and stuff
    if loss_fig:
        # Training and validation losses
        fig = plot_training(train_loss, valid_loss)
        fig.savefig(save_path / f'{name}_loss.png')
        # fig.show()
        plt.close(fig)

    model.eval()        
    return train_loss, valid_loss


def train_batch_iterate(model, train_data, val_data, n_epochs, learning_rate, ncond, path, name, rand_perm_target=False, inverse=False, loss_fig=True, device='cpu'):
    # try:
    #     train_data.paired() and val_data.paired()
    # except(AttributeError, TypeError):
    #   raise AssertionError('Training data should be a DataToData object')

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

        for step, pairdata in enumerate(train_data):
            model.train()
            if step % 2 == 0 + 1*int(inverse):
                data = pairdata[0]
                inv = False
            else:
                data = pairdata[1]
                inv = True

            optimizer.zero_grad()
            if ncond is not None:
                inputs, context_l, context_r = data[ddir]
                if rand_perm_target:
                    context_r = shuffle_tensor(context_l)
            else:
                inputs, context_l, context_r = data, None, None
            
            logprob = -model.log_prob(inputs, context_l=context_l, context_r=context_r, inverse=inv).mean()
            
            logprob.backward()
            optimizer.step()
            scheduler.step()
            t_loss.append(logprob.item())
        
        train_loss[epoch] = torch.tensor(t_loss).mean()
        v_loss = torch.zeros(len(val_data))
        for v_step, data in enumerate(val_data):
            for ddir in [0,1]:
                if ncond is not None:
                    inputs, context_l, context_r = data[ddir]
                    if rand_perm_target:
                        context_r = shuffle_tensor(context_l)
                else:
                    inputs, context_l, context_r = data[ddir], None, None

                with torch.no_grad():
                    v_loss[v_step] = -0.5*model.log_prob(inputs, context_l=context_l, context_r=context_r, inverse=ddir).mean()
        valid_loss[epoch] = v_loss.mean()

        torch.save(model.state_dict(), save_path / f'epoch_{epoch}_valloss_{valid_loss[epoch]:.3f}.pt')
        print(f"Loss = {train_loss[epoch]:.3f},\t val_loss = {valid_loss[epoch]:.3f}")

    ###insert saving of losses and plots and stuff
    if loss_fig:
        # Training and validation losses
        fig = plot_training(train_loss, valid_loss)
        fig.savefig(save_path / f'{name}_loss.png')
        # fig.show()
        plt.close(fig)

    model.eval()        
    return train_loss, valid_loss