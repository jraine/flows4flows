import pathlib

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from ffflows import distance_penalties
from ffflows.distance_penalties import AnnealedPenalty
from ffflows.models import DeltaFlowForFlow, ConcatFlowForFlow, DiscreteBaseFlowForFlow, \
    DiscreteBaseConditionFlowForFlow, NoContextFlowForFlow
from ffflows.data.plane import ConcentricRings, FourCircles, CheckerboardDataset, TwoSpiralsDataset, Star, \
    Anulus
from ffflows.data.conditional_plane import RotatedData, RadialScale, ElipseShift
from ffflows.utils import shuffle_tensor

from nflows import transforms
from nflows.utils import tensor2numpy

import pandas as pd
import numpy as np

from plot import plot_training
import matplotlib.pyplot as plt


def get_activation(name, *args, **kwargs):
    actdict = {
        "linear": lambda x: x,
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "sigmoid": F.sigmoid,
        "selu": F.selu,
        "celu": F.celu,
        "elu": F.elu,
        "swish": F.hardswish,
        "softplus": F.softplus,
    }
    assert name.lower() in actdict, f"Currently {name} is not supported.  Choose one of '{actdict.keys()}'"

    return actdict[name.lower()]


def get_data(name, num_points, *args, **kwargs):
    datadict = {
        "anulus": Anulus,
        "ring": Anulus,
        "concentricrings": ConcentricRings,
        "fourcircles": FourCircles,
        "checkerboard": CheckerboardDataset,
        "spirals": TwoSpiralsDataset,
        "star": Star,
        "eightstar": Star,
    }
    assert name.lower() in datadict.keys(), f"Currently {name} is not supported. Choose one of '{datadict.keys()}'"
    # batch_size = num_points if batch_size is None else batch_size
    if name.lower() == 'ring':
        return datadict[name.lower()](num_points, radius=1.25)
    elif name.lower == 'eightstar':
        return datadict[name.lower()](num_points, num_bars=8)
    else:
        return datadict[name.lower()](num_points)
    # return datadict[name.lower()](num_points)


def get_conditional_data(conditional_type, base_name, num_points, *args, **kwargs):
    base_data = get_data(base_name, num_points, *args, **kwargs)

    data_wrapper = {
        "rotation": RotatedData,
        "radial": RadialScale,
        "ellipse": ElipseShift
    }[conditional_type.lower()]
    return data_wrapper(base_data)


def set_penalty(f4flow, penalty, weight, anneal=False):
    if penalty not in ['None', None]:
        if penalty == 'l1':
            penalty_constr = distance_penalties.LOnePenalty
        elif penalty == 'l2':
            penalty_constr = distance_penalties.LTwoPenalty
        penalty = penalty_constr(weight)
        if anneal:
            penalty = AnnealedPenalty(penalty)
        f4flow.add_penalty(penalty)


def get_flow4flow_ncond(name):
    # TODO merge this function with the get_flow4flow
    f4fdict = {
        "delta": 1,
        "no_context": 1,
        "concat": 2,
        "discretebase": 1,
        "discretebasecondition": 1,
    }
    assert name.lower() in f4fdict, f"Currently {f4fdict} is not supported. Choose one of '{f4fdict.keys()}'"

    return f4fdict[name]


def get_flow4flow(name, *args, **kwargs):
    f4fdict = {
        "delta": DeltaFlowForFlow,
        "no_context": NoContextFlowForFlow,
        "concat": ConcatFlowForFlow,
        "discretebase": DiscreteBaseFlowForFlow,
        "discretebasecondition": DiscreteBaseConditionFlowForFlow,
    }
    assert name.lower() in f4fdict, f"Currently {f4fdict} is not supported. Choose one of '{f4fdict.keys()}'"

    return f4fdict[name](*args, **kwargs)


def spline_inn(inp_dim, nodes=128, num_blocks=2, num_stack=3, tail_bound=3.5, tails='linear', activation=F.relu, lu=0,
               num_bins=10, context_features=None, flow_for_flow=False):
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

    if not (flow_for_flow and (num_stack % 2 == 0)):
        # If the above conditions are satisfied then you want to permute back to the original ordering such that the
        # output features line up with their original ordering.
        transform_list = transform_list[:-1]

    return transforms.CompositeTransform(transform_list)


def train(model, train_data, val_data, n_epochs, learning_rate, ncond, path, name, rand_perm_target=False,
          inverse=False, loss_fig=True, device='cpu', gclip=None):
    save_path = pathlib.Path(path / name)
    save_path.mkdir(parents=True, exist_ok=True)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_steps = len(train_data) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, last_epoch=-1,
                                                           eta_min=0)
    if hasattr(model, 'distance_object'):
        if hasattr(model.distance_object, 'set_n_steps'):
            model.distance_object.set_n_steps(num_steps)
    train_loss = torch.zeros(n_epochs)
    valid_loss = torch.zeros(n_epochs)
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        t_loss = []

        for step, data in enumerate(train_data):
            model.train()

            optimizer.zero_grad()
            if ncond is not None:
                inputs, input_context = data[0].to(device), data[1].to(device)
                target_context = shuffle_tensor(input_context) if rand_perm_target else None
            else:
                inputs, input_context, target_context = data.to(device), None, None

            logprob = -model.log_prob(inputs, input_context=input_context, target_context=target_context,
                                      inverse=inverse).mean()

            logprob.backward()
            if gclip not in ['None', None]:
                clip_grad_norm_(model.parameters(), gclip)
            optimizer.step()
            scheduler.step()
            t_loss.append(logprob.item())

        train_loss[epoch] = torch.tensor(t_loss).mean()
        v_loss = torch.zeros(len(val_data))
        for v_step, data in enumerate(val_data):
            if ncond is not None:
                inputs, input_context = data[0].to(device), data[1].to(device)
                target_context = shuffle_tensor(input_context) if rand_perm_target else None
            else:
                inputs, input_context, target_context = data.to(device), None, None

            with torch.no_grad():
                v_loss[v_step] = -model.log_prob(inputs, input_context=input_context, target_context=target_context,
                                                 inverse=inverse).mean()
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


def train_batch_iterate(model, train_data, val_data, n_epochs, learning_rate, ncond, path, name, rand_perm_target=False,
                        inverse=False, loss_fig=True, device='cpu'):
    # try:
    #     train_data.paired() and val_data.paired()
    # except(AttributeError, TypeError):
    #   raise AssertionError('Training data should be a DataToData object')
    # TODO block this to reduce on code duplication
    save_path = pathlib.Path(path / name)
    save_path.mkdir(parents=True, exist_ok=True)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_steps = len(train_data) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, last_epoch=-1,
                                                           eta_min=0)
    if hasattr(model, 'distance_object.set_n_steps'):
        model.distance_object.set_n_steps(num_steps)
    train_loss = torch.zeros(n_epochs)
    valid_loss = torch.zeros(n_epochs)
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        t_loss = []

        for step, pairdata in enumerate(train_data):
            model.train()
            if step % 2 == 0 + 1 * int(inverse):
                data = pairdata[0].to(device)
                inv = False
            else:
                data = pairdata[1].to(device)
                inv = True

            optimizer.zero_grad()
            if ncond is not None:
                inputs, context_l, context_r = data[ddir][0].to(device), data[ddir][1].to(device), data[ddir][2].to(
                    device)
                if rand_perm_target:
                    context_r = shuffle_tensor(context_l)
            else:
                inputs, context_l, context_r = data.to(device), None, None

            logprob = -model.log_prob(inputs, input_context=context_l, target_context=context_r, inverse=inv).mean()

            logprob.backward()
            optimizer.step()
            scheduler.step()
            t_loss.append(logprob.item())

        train_loss[epoch] = torch.tensor(t_loss).mean()
        v_loss = torch.zeros(len(val_data))
        for v_step, data in enumerate(val_data):
            for ddir in [0, 1]:
                if ncond is not None:
                    inputs, context_l, context_r = data[ddir][0].to(device), data[ddir][1].to(device), data[ddir][2].to(
                        device)
                    if rand_perm_target:
                        context_r = shuffle_tensor(context_l)
                else:
                    inputs, context_l, context_r = data[ddir].to(device), None, None

                with torch.no_grad():
                    v_loss[v_step] = -0.5 * model.log_prob(inputs, input_context=context_l, target_context=context_r,
                                                           inverse=ddir).mean()
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

def tensor_to_str(tensor):
    '''Convert a tensor to a string or list of strings. Can be a tensor of shape (), (N,), (N,M), and any other squeezeable shapes'''

    ##TODO: Have this walk through all dims in the shape tensor to nested lists.
    ###Can squeeze first, then get shape
    def t_to_s(t):
        if t.view(1, -1).shape[1] > 1:
            return '_'.join([f'{a:.2f}' for a in t.squeeze()])
        else:
            return f'{t.squeeze():.2f}'

    if len(tensor.shape) < 2:
        return t_to_s(tensor)
    elif tensor.shape[0] > 1:
        return [t_to_s(t) for t in tensor]
    else:
        return t_to_s(tensor)

def dump_to_df(*args, col_names=None):
    data = [tensor2numpy(d) for d in args]
    if len(np.unique(lens := [len(d) for d in data])) != 1:
        print(f"Arrays not all same length, received f{lens}")
        exit(50)
    elif len(np.unique(shapes := [d.shape[:-1] for d in data])) != 1:
        print(f"Arrays not all same shape up until last axis, received f{shapes}")
        exit(51)
    data = np.concatenate(data,axis=-1)

    if col_names is not None:
        cols = col_names
    else:
        cols = [f'Array{i}' for i in range(len(args))]

    return pd.DataFrame(data,columns=cols)