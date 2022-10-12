from scipy.stats import binned_statistic_2d

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from nflows import transforms
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.utils import tensor2numpy
from torch.utils.data import DataLoader

from ffflows.data.dist_to_dist import UnconditionalDataToData
from ffflows.data.plane import Anulus, ConditionalAnulus, CheckerboardDataset, FourCircles, ConcentricRings, \
    TwoSpiralsDataset
from ffflows.models import FlowForFlow, DeltaFlowForFlow
from torch.nn import functional as F
import torch


# TODO package everythin up to avoid code duplication!
def spline_inn(inp_dim, nodes=128, num_blocks=2, nstack=3, tail_bound=3.5, tails='linear', activation=F.relu, lu=0,
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


def plot_training(training, validation):
    fig, ax = plt.subplots(1, 1)
    ax.plot(tensor2numpy(training), label='Training')
    ax.plot(tensor2numpy(validation), label='Validation')
    ax.legend()
    return fig


def train(model, train_loader, valid_loader, n_epochs, learning_rate, device, directory, sv_nm=None):
    directory.mkdir(exist_ok=True, parents=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_steps = len(train_loader) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, last_epoch=-1,
                                                           eta_min=0)
    train_loss = torch.zeros(n_epochs)
    valid_loss = torch.zeros(n_epochs)
    for epoch in range(n_epochs):
        t_loss = []
        # Redefine the loaders at the start of every epoch, this will also resample if resampling the mass
        for step, data in enumerate(train_loader):
            model.train()
            # Zero the accumulated gradients
            optimizer.zero_grad()
            # Get the loss
            loss = model.compute_loss(data)
            # Calculate the derivatives
            loss.backward()
            # Step the optimizers and schedulers
            optimizer.step()
            scheduler.step()
            # Store the loss
            t_loss += [loss.item()]
        # Save the model
        torch.save(model.state_dict(), directory / f'{epoch}')

        # Store the losses across the epoch and start the validation
        train_loss[epoch] = torch.tensor(t_loss).mean()
        v_loss = torch.zeros(len(valid_loader))
        for v_step, (data) in enumerate(valid_loader):
            with torch.no_grad():
                v_loss[v_step] = model.compute_loss(data)
        valid_loss[epoch] = v_loss.mean()

    if sv_nm is not None:
        # Training and validation losses
        fig = plot_training(train_loss, valid_loss)
        fig.savefig(sv_nm)
        # fig.show()
        plt.close(fig)

    model.eval()
    return train_loss, valid_loss


def plot_data(data, nm):
    data = tensor2numpy(data)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bound = 4
    bounds = [[-bound, bound], [-bound, bound]]
    ax.hist2d(data[:, 0], data[:, 1], bins=256)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    plt.savefig(nm)


# Need to be able to turn off gradients for the base density
def no_more_grads(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def make_colormap(plt=False):
    def arr_creat(upperleft, upperright, lowerleft, lowerright):
        arr = np.linspace(np.linspace(lowerleft, lowerright, arrwidth),
                          np.linspace(upperleft, upperright, arrwidth), arrheight, dtype=int)
        return arr[:, :, None]

    arrwidth = 256
    arrheight = 256

    r = arr_creat(0, 255, 0, 255)
    g = arr_creat(0, 0, 255, 0)
    b = arr_creat(255, 255, 0, 0)

    img = np.concatenate([r, g, b], axis=2)

    if plt:
        plt.imshow(img, origin="lower")
        plt.axis("off")

        plt.show()
    return img


def assign_colors(img, input_data):
    dt = tensor2numpy(input_data)
    bins = np.linspace(-4, 4, 256)
    _, _, _, color_ind = binned_statistic_2d(dt[:, 0], dt[:, 1], dt[:, 1], bins=(bins, bins),
                                             expand_binnumbers=True)
    return img[color_ind[0], color_ind[1]]


def add_scatter(ax, data, colors):
    dt = tensor2numpy(data)
    ax.scatter(dt[:, 0], dt[:, 1], s=0.1, c=colors / 256, alpha=0.8)


def plot_arrays(dict_of_data, sv_nm, colors=None):
    n_figs = len(dict_of_data)
    fig, ax = plt.subplots(1, n_figs, figsize=(6 * n_figs, 5))
    for i, (nm, data) in enumerate(dict_of_data.items()):
        if colors is None:
            img = make_colormap()
            colors = assign_colors(img, data)
        add_scatter(ax[i], data, colors)
        ax[i].set_title(nm)
    fig.savefig(sv_nm)


def move_dists():
    # Hyperparameters
    batch_size = 128

    # Experiment path
    top_save = Path('results/dist_to_dist')
    top_save.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define data
    n_points = int(1e4)
    batch_size = 128
    valid_batch_size = 128

    def make_data(n_points):
        # TODO can the flow learn the identity? Or does it try to shift things around?
        left_dataset = CheckerboardDataset(num_points=n_points)
        right_dataset = FourCircles(num_points=n_points)
        right_dataset = ConcentricRings(num_points=n_points)
        right_dataset = TwoSpiralsDataset(num_points=n_points)
        # right_dataset = CheckerboardDataset(num_points=n_points)
        return UnconditionalDataToData(left_dataset, right_dataset)

    train_dataset = make_data(n_points)
    valid_dataset = make_data(n_points)

    class UnconditionalFlow(Flow):
        """For the demo the way the conditioning is done is hard coded."""

        def compute_loss(self, data):
            return -self.log_prob(data).mean()

    # Define the base density
    left_base_density = UnconditionalFlow(spline_inn(2), StandardNormal([2]))
    right_base_density = UnconditionalFlow(spline_inn(2), StandardNormal([2]))

    # Train the base density
    def train_base(density, func_direct):
        models_save = top_save / f'base_densities_{func_direct}'
        models_save.mkdir(exist_ok=True)
        left_loader_train = DataLoader(dataset=getattr(train_dataset, func_direct)(), batch_size=batch_size)
        left_loader_valid = DataLoader(dataset=getattr(valid_dataset, func_direct)(), batch_size=batch_size)
        train(density, left_loader_train, left_loader_valid, 30, 0.001, device, models_save,
              top_save / f'loss_base_{func_direct}.png')

    train_base(left_base_density, 'left')
    train_base(right_base_density, 'right')

    # Evaluate the base density
    def eval_base(base_density, direction='left'):
        with torch.no_grad():
            samples = base_density.sample(int(1e5))
        plot_data(samples, top_save / f'base_density_{direction}.png')

    eval_base(left_base_density, 'left')
    eval_base(right_base_density, 'right')
    [no_more_grads(density) for density in (left_base_density, right_base_density)]

    # # Define the flow for flow object

    class fff(FlowForFlow):
        """Try and test with current setup"""

        def __init__(self, *args, **kwargs):
            super(fff, self).__init__(*args, **kwargs)
            self.iter = 0

        def context_func(self, context_l, context_r):
            return None

        def _direction_func(self, context_l, context_r):
            return None

        def compute_loss(self, data):
            left_data, right_data = data
            if self.iter == 0:
                self.iter = 1
                return -self.log_prob(left_data, inverse=True).mean()
            else:
                self.iter = 0
                return -self.log_prob(right_data, inverse=False).mean()

        def final_eval(self, data, target_shift):
            data, context = self.split_data(data)
            return self.transform(data, context, context + target_shift * torch.ones_like(context))

    flow_for_flow = fff(spline_inn(2, context_features=1), left_base_density, right_base_density)

    # Train the flow for flow model
    models_save = top_save / 'f4fs'
    models_save.mkdir(exist_ok=True)
    train_loader = DataLoader(dataset=train_dataset.paired(), batch_size=batch_size)
    valid_loader = DataLoader(dataset=valid_dataset.paired(), batch_size=batch_size)
    train(flow_for_flow, train_loader, valid_loader, 30, 0.001, device, models_save, top_save / 'loss_f4f.png')

    n_points = int(1e6)
    n_points = int(1e5)
    test_data = make_data(n_points)

    input_data = test_data.left().data
    plot_data(input_data, top_save / f'flow_for_flow_input.png')
    plot_data(test_data.right().data, top_save / f'flow_for_flow_target.png')
    left_to_right, _ = flow_for_flow.transform(input_data, inverse=True)
    plot_data(left_to_right, top_save / f'left_to_right.png')

    left_bd_enc = flow_for_flow.base_flow_fwd.transform_to_noise(input_data)
    right_bd_dec, _ = flow_for_flow.base_flow_inv._transform.inverse(left_bd_enc)
    plot_arrays({
        'Input Data': input_data,
        'FFF': left_to_right,
        'BdTransfer': right_bd_dec
    }, top_save / 'colored_lr.png')


if __name__ == '__main__':
    move_dists()
