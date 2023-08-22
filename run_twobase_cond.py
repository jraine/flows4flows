import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib
from ffflows.models import BaseFlow
from ffflows.utils import set_trainable

import torch
from torch.utils.data import DataLoader

from nflows.distributions import StandardNormal

from ffflows.utils import get_activation, get_data, get_flow4flow, train, train_batch_iterate, spline_inn, set_penalty, \
    dump_to_df, get_conditional_data, tensor_to_str
import matplotlib.pyplot as plt
from plot import plot_training, plot_data, plot_arrays

from ffflows.data.dist_to_dist import PairedConditionalDataToTarget

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


def get_datasets(cfg):
    n_points = int(cfg.general.n_points)
    condition_type = cfg.general.condition
    return [get_conditional_data(condition_type, bd_conf.data, n_points) for bd_conf in
            [cfg.base_dist.left, cfg.base_dist.right]]


@hydra.main(version_base=None, config_path="conf/", config_name="cond_twobase")
def main(cfg: DictConfig) -> None:
    print("Configuring job with following options")
    print(OmegaConf.to_yaml(cfg))
    outputpath = pathlib.Path(cfg.output.save_dir + '/' + cfg.output.name)
    outputpath.mkdir(parents=True, exist_ok=True)
    with open(outputpath / f"{cfg.output.name}.yaml", 'w') as file:
        OmegaConf.save(config=cfg, f=file)

    if cfg.general.ncond is None or cfg.general.ncond < 1:
        print(f"Expecting conditions, {cfg.general.ncond} was passed as the number of conditions.")
        exit(42)

    # Set device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Get training data
    n_points = int(cfg.general.n_points)
    condition_type = cfg.general.condition
    ncond_base = None if cfg.general.ncond == 0 else cfg.general.ncond
    base_data_l, base_data_r = [DataLoader(dataset=get_conditional_data(condition_type, bd_conf.data, n_points),
                                           batch_size=bd_conf.batch_size, shuffle=True) \
                                for bd_conf in [cfg.base_dist.left, cfg.base_dist.right]]
    val_base_data_l, val_base_data_r = [
        DataLoader(dataset=get_conditional_data(condition_type, bd_conf.data, n_points),
                   batch_size=1000, shuffle=True) \
        for bd_conf in [cfg.base_dist.left, cfg.base_dist.right]]

    plot_data(get_data(cfg.base_dist.left.data, n_points).data,
              outputpath / f'base_density_left_data.png')
    plot_data(get_data(cfg.base_dist.right.data, n_points).data,
              outputpath / f'base_density_right_data.png')

    # Train base
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

        set_trainable(base_flow, False)


        if cfg.base_dist.plot:
            base_flow.to(device)

            nevalpoints = 6
            bd_samples = []
            bd_path = outputpath / f'base_{label}' / 'evaluation'
            bd_path.mkdir(exist_ok=True, parents=True)
            with torch.no_grad():
                for right_cond in (
                        evals := get_conditional_data(condition_type, bd_conf.data, n_points).get_default_eval(
                            nevalpoints)):
                    nsamples = int(1e5)
                    right_cond = torch.Tensor([right_cond]).view(1, -1).to(device)
                    plot_data(
                        sampled := base_flow.sample(nsamples, context=right_cond, batch_size=int(1e5)).view(-1, 2),
                        bd_path / f'base_density_{tensor_to_str(right_cond)}.png'
                    )
                    bd_samples.append(sampled)
            df = dump_to_df(*bd_samples,
                            col_names=[f'cond_{ev:.2f}_{coord}'.replace('.', '_') for ev in evals for coord in
                                       ['x', 'y']])
            df.to_hdf(bd_path / 'eval_data.h5', f'base_dist')

    # Train Flow4Flow
    
    f4flow = get_flow4flow('discretebasecondition',
                           spline_inn(cfg.general.data_dim,
                                      nodes=cfg.top_transformer.nnodes,
                                      num_blocks=cfg.top_transformer.nblocks,
                                      num_stack=cfg.top_transformer.nstack,
                                      tail_bound=4.0,
                                      activation=get_activation(cfg.top_transformer.activation),
                                      num_bins=cfg.top_transformer.nbins,
                                      context_features=ncond_base,
                                      flow_for_flow=True,
                                      identity_init = cfg.top_transformer.identity_init
                                      ),
                           distribution_right=base_flow_r,
                           distribution_left=base_flow_l)

    set_penalty(f4flow, cfg.top_transformer.penalty, cfg.top_transformer.penalty_weight, cfg.top_transformer.anneal)

    train_data = PairedConditionalDataToTarget(*get_datasets(cfg))
    val_data = PairedConditionalDataToTarget(*get_datasets(cfg))
    
    print("Training additions for Flow4Flow model:")
    if cfg.top_transformer.identity_init:
        print("\tModel initialized to the identity.")
    if cfg.top_transformer.penalty not in [None, "None"]:
        print(f"\tModel trained with {cfg.top_transformer.penalty} loss with weight {cfg.top_transformer.penalty_weight}.")
    if (not cfg.top_transformer.identity_init) and (cfg.top_transformer.penalty in [None, "None"]):
        print("\tNone.")

    if pathlib.Path(cfg.top_transformer.load_path).is_file():
        print(f"Loading Flow4Flow from model: {cfg.top_transformer.load_path}")
        f4flow.load_state_dict(torch.load(cfg.top_transformer.load_path, map_location=device))

    elif ((direction := cfg.top_transformer.direction.lower()) == 'iterate'):
        print("Training Flow4Flow model iteratively")
        iteration_steps = cfg.top_transformer.iteration_steps if 'iteration_steps' in cfg.top_transformer else 1
        train_f4f_iterate(f4flow, train_data, val_data, cfg.top_transformer.batch_size,
                          cfg.top_transformer.nepochs, cfg.top_transformer.lr, ncond_base,
                          outputpath, iteration_steps=iteration_steps,
                          name='f4f', device=device, gclip=cfg.top_transformer.gclip)

    elif (direction == 'alternate'):
        print("Training Flow4Flow model alternating every batch")
        train_batch_iterate(f4flow, DataLoader(train_data.paired(), batch_size=cfg.top_transformer.batch_size,
                                               shuffle=True),
                            DataLoader(val_data.paired(), batch_size=cfg.top_transformer.batch_size),
                            cfg.top_transformer.nepochs, cfg.top_transformer.lr, ncond_base,
                            outputpath, name='f4f', device=device, gclip=cfg.top_transformer.gclip)

    else:
        if (direction == 'forward' or direction == 'both'):
            print("Training Flow4Flow model forwards")
            train_f4f_forward(f4flow,
                              DataLoader(train_data.left(), batch_size=cfg.top_transformer.batch_size, shuffle=True),
                              DataLoader(val_data.left(), batch_size=1000),
                              cfg.top_transformer.nepochs, cfg.top_transformer.lr, ncond_base,
                              outputpath, name='f4f_fwd', device=device, gclip=cfg.top_transformer.gclip)

        if (direction == 'inverse' or direction == 'both'):
            print("Training Flow4Flow model backwards")
            train_f4f_inverse(f4flow,
                              DataLoader(train_data.right(), batch_size=cfg.top_transformer.batch_size, shuffle=True),
                              DataLoader(val_data.right(), batch_size=1000),
                              cfg.top_transformer.nepochs, cfg.top_transformer.lr, ncond_base,
                              outputpath, name='f4f_inv', device=device, gclip=cfg.top_transformer.gclip)

    with torch.no_grad():
        f4flow.to(device)

        # Colored/conditional plots
        test_data = get_conditional_data(condition_type, cfg.base_dist.left.data, n_points)

        # This will return a set of conditions to map to, and ensure test_data contains points from the same condition
        test_points = test_data.get_default_eval(6)
        flow4flow_dir = outputpath / 'flow4flow_plots'
        flow4flow_dir.mkdir(exist_ok=True, parents=True)
        debug_dir = flow4flow_dir / 'debug'
        debug_dir.mkdir(exist_ok=True, parents=True)
        for con in test_points:
            # Handle the broadcasting
            # TODO this isn't generic across the different conditional datasets
            if condition_type == "rotation":
                con *= test_data.max_angle
            elif condition_type == "radial":
                con *= test_data.max_scale
            else:
                print("ERROR: not implemented")
            left_data, left_cond = test_data._get_conditional(con.item())
            left_data = torch.Tensor(left_data).to(device)
            left_cond = (left_cond * torch.ones(len(left_data), 1)).to(device)
            right_cond = left_cond

            # Transform the data
            transformed, _ = f4flow.batch_transform(left_data, left_cond, right_cond, batch_size=1000)
            # Plot the output densities
            plot_data(transformed, flow4flow_dir / f'flow_for_flow_{tensor_to_str(con)}.png')
            # Get the transformation that results from going via the base distributions
            left_bd_enc = f4flow.base_flow_left.transform_to_noise(left_data, left_cond)
            right_bd_dec, _ = f4flow.base_flow_right._transform.inverse(left_bd_enc, right_cond.view(-1, 1))
            # Plot how each point is shifted
            plot_arrays({
                'Input Data': left_data,
                'FFF': transformed,
                'BdTransfer': right_bd_dec
            }, flow4flow_dir, f'{con.item():.2f}')
            plot_data(transformed, debug_dir / f'transformed_density_{tensor_to_str(right_cond[0])}.png')
            plot_data(right_bd_dec, debug_dir / f'bd_transformed_density_{tensor_to_str(right_cond[0])}.png')

            ##dump data
            df = dump_to_df(left_data, left_cond, right_cond * torch.ones_like(left_cond), transformed, left_bd_enc,
                            right_bd_dec,
                            col_names=['input_x', 'input_y', 'left_cond', 'right_cond',
                                       'transformed_x', 'transformed_y', 'left_enc_x', 'left_enc_y',
                                       'base_transfer_x', 'base_transfer_y'])
            df.to_hdf(flow4flow_dir / 'eval_data_conditional.h5', f'f4f_{con.item():.2f}'.replace('.', '_'))


if __name__ == "__main__":
    main()
