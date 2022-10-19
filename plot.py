import matplotlib.pyplot as plt
from nflows.utils import tensor2numpy
import numpy as np
from scipy.stats import binned_statistic_2d


def assign_colors(img, input_data):
    dt = tensor2numpy(input_data)
    bins = np.linspace(-4, 4, 256)
    _, _, _, color_ind = binned_statistic_2d(dt[:, 0], dt[:, 1], dt[:, 1], bins=(bins, bins),
                                             expand_binnumbers=True)

    return img[color_ind[0], color_ind[1]]

def add_scatter(ax, data, colors):
    dt = tensor2numpy(data)
    ax.scatter(dt[:, 0], dt[:, 1], s=0.1, c=colors / 256, alpha=0.8)

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

def plot_data(data, nm):
    data = tensor2numpy(data)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bound = 4
    bounds = [[-bound, bound], [-bound, bound]]
    ax.hist2d(data[:, 0], data[:, 1], bins=256)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    plt.savefig(nm)

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