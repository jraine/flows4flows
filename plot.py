import matplotlib
import pandas as pd

from utils import get_numpy_data

matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.switch_backend('agg')
# import pandas as pd
from nflows.utils import tensor2numpy
import numpy as np
from scipy.stats import binned_statistic_2d


def assign_colors(img, input_data):
    dt = tensor2numpy(input_data)
    bins = np.linspace(-4, 4, 255)
    _, _, _, color_ind = binned_statistic_2d(dt[:, 0], dt[:, 1], dt[:, 1], bins=(bins, bins),
                                             expand_binnumbers=True)
    return img[color_ind[0], color_ind[1]]


def add_scatter(ax, data, colors, s=0.1):
    dt = tensor2numpy(data)
    ax.scatter(dt[:, 0], dt[:, 1], s=s, c=colors / 256, alpha=0.8)


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


def plot_training(training, validation):
    fig, ax = plt.subplots(1, 1)
    ax.plot(tensor2numpy(training), label='Training')
    ax.plot(tensor2numpy(validation), label='Validation')
    ax.legend()
    return fig


def set_bounds(ax, bound=4):
    bounds = [[-bound, bound], [-bound, bound]]
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])


def plot_data(data, nm):
    data = tensor2numpy(data)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist2d(data[:, 0], data[:, 1], bins=256)
    set_bounds(ax)
    plt.savefig(nm)


def plot_arrays(dict_of_data, sv_dir, sv_nm, colors=None):
    n_figs = len(dict_of_data)
    fig, ax = plt.subplots(1, n_figs, figsize=(6 * n_figs, 5))
    for i, (nm, data) in enumerate(dict_of_data.items()):
        if colors is None:
            img = make_colormap()
            colors = assign_colors(img, data)
        add_scatter(ax[i], data, colors)
        ax[i].set_title(nm)
        set_bounds(ax[i])
    fig.savefig(sv_dir / f'colored_{sv_nm}.png')

    # # TODO want to do something like this but pandas is broken in the container?
    # data = {k: tensor2numpy(x) for k, x in dict_of_data.items()}
    # ln, r = data[list(data.keys())[0]].shape
    # df = pd.DataFrame({k: x.ravel() for k, x in data.items()},
    #                   index=pd.MultiIndex.from_product([np.arange(ln), np.arange(r)]))
    # df.to_csv(sv_dir / f'{sv_nm}.csv', index=False)


def plot_grid(grid, columns, nm, n_points=int(1e4)):
    """
    Plot a grid of figures showing inputs to outputs
    :param grid: A dictionary with keys input_to_target indexing paths to hdf5 saved pandas dataframes.
    :param columns: The columns of the data (loaded from the above dataframe) to plot.
    :param nm: Name with which to save the plot
    :param n_points: Number of points to sample for showing the input/target distributions
    :return:
    """
    inps = []
    trgts = []
    for entry in grid:
        i, _, t = entry.split('_')
        inps += [i]
        trgts += [t]

    inps = np.unique(inps)
    trgts = np.unique(trgts)

    def add_2d_hist(axis, data):
        axis.hist2d(data[:, 0], data[:, 1], bins=256)
        set_bounds(axis)

    N_inputs = len(inps)
    N_targets = len(trgts)

    # Add one because we want to plot the data around the perimeter.
    fig, ax = plt.subplots(N_inputs + 1, N_targets + 1,
                           figsize=(5 * (N_targets + 1), 5 * (N_inputs + 1)))
    fig.delaxes(fig.axes[0])

    for i, inp in enumerate(inps):
        data = get_numpy_data(inp, n_points)
        add_2d_hist(ax[i + 1, 0], data)

    for i, inp in enumerate(trgts):
        data = get_numpy_data(inp, n_points)
        add_2d_hist(ax[0, i + 1], data)

    for i, inp in enumerate(inps):
        for j, trgt in enumerate(trgts):
            data = pd.read_hdf(grid[f'{inp}_to_{trgt}'])[columns].to_numpy()
            add_2d_hist(ax[i + 1, j + 1], data)

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Input", fontsize=40)
    plt.title("Target", fontsize=40, pad=30)

    fig.tight_layout()
    fig.savefig(nm)
