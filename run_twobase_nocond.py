import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib
import glob
import os
import ffflows
from ffflows import distance_penalties
from ffflows.models import BaseFlow
from ffflows.utils import set_trainable

import torch
from torch.utils.data import DataLoader

from nflows import transforms
from nflows.distributions import StandardNormal
from nflows.flows import Flow

from utils import get_activation, get_data, get_flow4flow, train, train_batch_iterate, spline_inn, set_penalty, \
    dump_to_df
import matplotlib.pyplot as plt
from plot import plot_training, plot_data, plot_arrays

from ffflows.data.dist_to_dist import UnconditionalDataToData

import numpy as np

np.random.seed(42)
torch.manual_seed(42)


def train_base(*args, **kwargs):
    return train(*args, **kwargs)


def train_f4f_forward(*args, **kwargs):
    return train(*args, **kwargs, rand_perm_target=True, inverse=False)


def train_f4f_inverse(*args, **kwargs):
    return train(*args, **kwargs, rand_perm_target=True, inverse=True)


def train_f4f_iterate(model, train_dataset, val_dataset, batch_size,
                      n_epochs, learning_rate, ncond, path, name,
                      iteration_steps=1,
                      rand_perm_target=False, inverse=False, loss_fig=True, device='cpu', gclip=None):
    loss_fwd = torch.zeros(n_epochs)
    val_loss_fwd = torch.zeros(n_epochs)
    loss_inv = torch.zeros(n_epochs)
    val_loss_inv = torch.zeros(n_epochs)

    for step in range((steps := n_epochs // iteration_steps)):
        print(f"Iteration {step + 1}/{steps}")
        for train_data, val_data, loss, val_loss, ddir, inv in zip([train_dataset.left(), train_dataset.right()],
                                                                   [val_dataset.left(), val_dataset.right()],
                                                                   [loss_fwd, loss_inv],
                                                                   [val_loss_fwd, val_loss_inv],
                                                                   ['fwd', 'inv'],
                                                                   [True, False]):
            print(("Forward" if ddir == 'fwd' else "Inverse"))
            loss_step, val_loss_step = train(model, DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True),
                                             DataLoader(dataset=val_data, batch_size=1000), iteration_steps,
                                             learning_rate, ncond, path, f'{name}_{ddir}_step_{step}',
                                             rand_perm_target=rand_perm_target, inverse=inv,
                                             loss_fig=False, device=device, gclip=gclip)
            loss[step * iteration_steps:(step + 1) * iteration_steps] = loss_step
            val_loss[step * iteration_steps:(step + 1) * iteration_steps] = val_loss_step

    if loss_fig:
        for loss, val_loss, ddir in zip([loss_fwd, loss_inv],
                                        [val_loss_fwd, val_loss_inv],
                                        ['fwd', 'inv']):
            fig = plot_training(loss, val_loss)
            fig.savefig(path / f'{name}_{ddir}_loss.png')
            # fig.show()
            plt.close(fig)

    model.eval()


@hydra.main(version_base=None, config_path="conf/", config_name="defaults_twobase")
def main(cfg: DictConfig) -> None:
    print("Configuring job with following options")
    print(OmegaConf.to_yaml(cfg))
    outputpath = pathlib.Path(cfg.output.save_dir + '/' + cfg.output.name)
    outputpath.mkdir(parents=True, exist_ok=True)
    with open(outputpath / f"{cfg.output.name}.yaml", 'w') as file:
        OmegaConf.save(config=cfg, f=file)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get training data
    n_points = int(cfg.general.npoints)
    base_data_l, base_data_r = [DataLoader(dataset=get_data(bd_conf.data, n_points),
                                           batch_size=bd_conf.batch_size,
                                           shuffle=True) \
                                for bd_conf in [cfg.base_dist.left, cfg.base_dist.right]]
    val_base_data_l, val_base_data_r = [DataLoader(dataset=get_data(bd_conf.data, n_points),
                                                   batch_size=1000,
                                                   shuffle=True) \
                                        for bd_conf in [cfg.base_dist.left, cfg.base_dist.right]]

    ncond_base = None if cfg.general.ncond == 0 else cfg.general.ncond
    ncond_f4f = ncond_base * 2 if ncond_base is not None else None

    plot_data(get_data(cfg.base_dist.left.data, n_points).data,
              outputpath / f'base_density_left_data.png')

    # Train base1
    base_flow_l, base_flow_r = [BaseFlow(spline_inn(cfg.general.data_dim,
                                                    nodes=bd_conf.nnodes,
                                                    num_blocks=bd_conf.nblocks,
                                                    num_stack=bd_conf.nstack,
                                                    tail_bound=4.0,
                                                    activation=get_activation(bd_conf.activation),
                                                    num_bins=bd_conf.nbins,
                                                    context_features=ncond_base
                                                    ),
                                         StandardNormal([cfg.general.data_dim])
                                         ) for bd_conf in [cfg.base_dist.left, cfg.base_dist.right]
                                ]

    for label, base_data, val_base_data, bd_conf, base_flow in zip(['left', 'right'],
                                                                   [base_data_l, base_data_r],
                                                                   [val_base_data_l, val_base_data_r],
                                                                   [cfg.base_dist.left, cfg.base_dist.right],
                                                                   [base_flow_l, base_flow_r]):

        if pathlib.Path(bd_conf.load_path).is_file():
            print(f"Loading base_{label} from model: {bd_conf.load_path}")
            base_flow.load_state_dict(torch.load(bd_conf.load_path, map_location=device))
        else:
            print(f"Training base_{label} distribution")
            train_base(base_flow, base_data, val_base_data,
                       bd_conf.nepochs, bd_conf.lr, ncond_base,
                       outputpath, name=f'base_{label}', device=device, gclip=cfg.base_dist.left.gclip)
            with open(outputpath / f'base_{label}' / f'{bd_conf.data.lower()}.yaml', 'w') as file:
                models =  glob.glob(str((outputpath / f'base_{label}' / 'epoch*pt').resolve()))
                models.sort(key=os.path.getmtime)
                bd_conf.load_path = models[-1]
                OmegaConf.save(config=bd_conf, f=file)

        set_trainable(base_flow, False)

        plot_data(base_flow.sample(int(1e5)), outputpath / f'base_density_{label}_samples.png')

    # Train Flow4Flow
    f4flow = get_flow4flow(cfg.top_transformer.flow4flow,
                           spline_inn(cfg.general.data_dim,
                                      nodes=cfg.top_transformer.nnodes,
                                      num_blocks=cfg.top_transformer.nblocks,
                                      num_stack=cfg.top_transformer.nstack,
                                      tail_bound=4.0,
                                      activation=get_activation(cfg.top_transformer.activation),
                                      num_bins=cfg.top_transformer.nbins,
                                      context_features=ncond_f4f,
                                      flow_for_flow=True
                                      ),
                           distribution_right=base_flow_r,
                           distribution_left=base_flow_l)

    set_penalty(f4flow, cfg.top_transformer.penalty, cfg.top_transformer.penalty_weight, cfg.top_transformer.anneal)

    train_data = UnconditionalDataToData(get_data(cfg.base_dist.left.data, n_points),
                                         get_data(cfg.base_dist.right.data, n_points))  # \
    # if ncond_f4f is None \
    # else ConditionalDataToData(get_data(cfg.base_dist.left.data, n_points),
    #                           get_data(cfg.base_dist.right.data, n_points))
    val_data = UnconditionalDataToData(get_data(cfg.base_dist.left.data, n_points),
                                       get_data(cfg.base_dist.right.data, n_points))  # \
    # if ncond_f4f is None \
    # else ConditionalDataToData(get_data(cfg.base_dist.left.data, n_points),
    #                           get_data(cfg.base_dist.right.data, n_points))

    if pathlib.Path(cfg.top_transformer.load_path).is_file():
        print(f"Loading Flow4Flow from model: {cfg.top_transformer.load_path}")
        f4flow.load_state_dict(torch.load(cfg.top_transformer.load_path, map_location=device))

    elif ((direction := cfg.top_transformer.direction.lower()) == 'iterate'):
        print("Training Flow4Flow model iteratively")
        iteration_steps = cfg.top_transformer.iteration_steps if 'iteration_steps' in cfg.top_transformer else 1
        train_f4f_iterate(f4flow, train_data, val_data, cfg.top_transformer.batch_size,
                          cfg.top_transformer.nepochs, cfg.top_transformer.lr, ncond_f4f,
                          outputpath, iteration_steps=iteration_steps,
                          name='f4f', device=device, gclip=cfg.top_transformer.gclip)

    elif (direction == 'alternate'):
        print("Training Flow4Flow model alternating every batch")
        train_batch_iterate(f4flow, DataLoader(train_data.paired(), batch_size=cfg.top_transformer.batch_size,
                                               shuffle=True),
                            DataLoader(val_data.paired(), batch_size=cfg.top_transformer.batch_size),
                            cfg.top_transformer.nepochs, cfg.top_transformer.lr, ncond_f4f,
                            outputpath, name='f4f', device=device, gclip=cfg.top_transformer.gclip) 

    else:
        if (direction == 'forward' or direction == 'both'):
            print("Training Flow4Flow model forwards")
            train_f4f_forward(f4flow,
                              DataLoader(train_data.left(), batch_size=cfg.top_transformer.batch_size, shuffle=True),
                              DataLoader(val_data.left(), batch_size=1000),
                              cfg.top_transformer.nepochs, cfg.top_transformer.lr, ncond_f4f,
                              outputpath, name='f4f_fwd', device=device, gclip=cfg.top_transformer.gclip)

        if (direction == 'inverse' or direction == 'both'):
            print("Training Flow4Flow model backwards")
            train_f4f_inverse(f4flow,
                              DataLoader(train_data.right(), batch_size=cfg.top_transformer.batch_size, shuffle=True),
                              DataLoader(val_data.right(), batch_size=1000),
                              cfg.top_transformer.nepochs, cfg.top_transformer.lr, ncond_f4f,
                              outputpath, name='f4f_inv', device=device, gclip=cfg.top_transformer.gclip)

    with torch.no_grad():
        f4flow.to(device)
        test_data = UnconditionalDataToData(get_data(cfg.base_dist.left.data, n_points),
                                            get_data(cfg.base_dist.right.data, n_points))

        left_data = test_data.left().data.to(device)
        right_data = test_data.right().data.to(device)

        plot_data(left_data, outputpath / f'flow_for_flow_left_input.png')
        plot_data(right_data, outputpath / f'flow_for_flow_right_input.png')
        left_to_right, _ = f4flow.batch_transform(left_data, inverse=False, batch_size=1000)
        plot_data(left_to_right, outputpath / f'left_to_right_transform.png')
        right_to_left, _ = f4flow.batch_transform(right_data, inverse=True, batch_size=1000)
        plot_data(right_to_left, outputpath / f'right_to_left_transform.png')
        sample_left = f4flow.base_flow_left.sample(n_points)
        plot_data(sample_left, outputpath / f'f4f_left_sample.png')
        sample_to_right, _ = f4flow.batch_transform(sample_left, inverse=False, batch_size=1000)
        plot_data(sample_to_right, outputpath / f'f4f_sample_left_transform_right.png')
        sample_right = f4flow.base_flow_right.sample(n_points)
        plot_data(sample_right, outputpath / f'f4f_right_sample.png')
        sample_to_left, _ = f4flow.batch_transform(sample_right, inverse=True, batch_size=1000)
        plot_data(sample_to_left, outputpath / f'f4f_sample_right_transform_left.png')

        left_bd_enc = f4flow.base_flow_left.transform_to_noise(left_data)
        right_bd_dec, _ = f4flow.base_flow_right._transform.inverse(left_bd_enc)
        plot_arrays({
            'Input Data': left_data,
            'FFF': left_to_right,
            'BdTransfer': right_bd_dec
        }, outputpath, 'left_to_right.png')

        df = dump_to_df(left_data, right_data, left_to_right, right_to_left,
                        sample_to_right, sample_to_left, left_bd_enc, right_bd_dec,
                        col_names=[f'{name}_{coord}' for name in ['left_data','right_data','left_to_right','right_to_left',
                                'sample_to_right','sample_to_left','left_enc', 'base_transfer'] for coord in ['x','y'] ])
        df.to_hdf(outputpath / 'eval_data.h5', 'f4f')


if __name__ == "__main__":
    main()
