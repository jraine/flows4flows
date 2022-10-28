import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib
from ffflows.models import BaseFlow
from ffflows.utils import set_trainable

import torch
from torch.utils.data import DataLoader

from nflows.distributions import StandardNormal

from utils import get_activation, get_data, get_flow4flow, train, spline_inn, get_conditional_data, tensor_to_str, \
    set_penalty, get_flow4flow_ncond, dump_to_df
from plot import plot_data, plot_arrays

from ffflows.data.dist_to_dist import ConditionalDataToData, ConditionalDataToTarget

import numpy as np

np.random.seed(42)
torch.manual_seed(42)


def train_base(*args, **kwargs):
    return train(*args, **kwargs, rand_perm_target=False)


def train_f4f(*args, **kwargs):
    return train(*args, **kwargs, rand_perm_target=True)


@hydra.main(version_base=None, config_path="conf/", config_name="cond_jointbase_default")
def main(cfg: DictConfig) -> None:
    print("Configuring job with following options")
    print(OmegaConf.to_yaml(cfg))
    outputpath = pathlib.Path(cfg.output.save_dir + '/' + cfg.output.name)
    outputpath.mkdir(parents=True, exist_ok=True)
    with open(outputpath / f"{cfg.output.name}.yaml", 'w') as file:
        OmegaConf.save(config=cfg, f=file)

    if cfg.general.ncond is None or cfg.general.ncond < 1:
        print(
            f"Cannot train Flows4Flows on the same base distribution without any conditions. You specified cfg.general.ncond = {cfg.general.ncond}. Exiting now.")
        exit(42)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get training data
    def get_data(n_points):
        return get_conditional_data(cfg.base_dist.condition, cfg.base_dist.base_data, n_points)

    n_points = int(cfg.general.n_points)
    base_data = DataLoader(
        dataset=get_data(n_points),
        batch_size=cfg.base_dist.batch_size,
        shuffle=True
    )
    val_base_data = DataLoader(
        dataset=get_data(n_points),
        batch_size=1000
    )

    # Train base
    base_flow = BaseFlow(spline_inn(cfg.general.data_dim,
                                    nodes=cfg.base_dist.nnodes,
                                    num_blocks=cfg.base_dist.nblocks,
                                    num_stack=cfg.base_dist.nstack,
                                    tail_bound=4.0,
                                    activation=get_activation(cfg.base_dist.activation),
                                    num_bins=cfg.base_dist.nbins,
                                    context_features=cfg.general.ncond
                                    ),
                         StandardNormal([cfg.general.data_dim])
                         )
    if pathlib.Path(cfg.base_dist.load_path).is_file():
        print(f"Loading base from model: {cfg.base_dist.load_path}")
        base_flow.load_state_dict(torch.load(cfg.base_dist.load_path, map_location=device))
    else:
        print("Training base distribution")
        train_base(base_flow, base_data, val_base_data,
                   cfg.base_dist.nepochs, cfg.base_dist.lr, cfg.general.ncond,
                   outputpath, name='base', device=device, gclip=cfg.base_dist.gclip)

    set_trainable(base_flow, False)

    base_flow.to(device)

    nevalpoints = 6
    bd_samples = []
    with torch.no_grad():
        for right_cond in (evals := get_data(20).get_default_eval(nevalpoints)):
            nsamples = int(1e6)
            right_cond = torch.Tensor([right_cond]).view(1, -1).to(device)
            plot_data(sampled := base_flow.sample(nsamples, context=right_cond, batch_size=int(1e5)).view(-1, 2),
                      outputpath / f'base_density_{tensor_to_str(right_cond)}.png')
            bd_samples.append(sampled)
    df = dump_to_df(*bd_samples,
                    col_names=[f'cond_{ev:.2f}_{coord}'.replace('.', '_') for ev in evals for coord in ['x', 'y']])
    df.to_hdf(outputpath / 'eval_data.h5', f'base_dist')

    # Train Flow4Flow
    n_cond = get_flow4flow_ncond(cfg.top_transformer.flow4flow) * cfg.general.ncond
    f4flow = get_flow4flow(cfg.top_transformer.flow4flow,
                           spline_inn(cfg.general.data_dim,
                                      nodes=cfg.top_transformer.nnodes,
                                      num_blocks=cfg.top_transformer.nblocks,
                                      num_stack=cfg.top_transformer.nstack,
                                      tail_bound=4.0,
                                      activation=get_activation(cfg.top_transformer.activation),
                                      num_bins=cfg.top_transformer.nbins,
                                      context_features=n_cond,
                                      flow_for_flow=True
                                      ),
                           base_flow)
    set_penalty(f4flow, cfg.top_transformer.penalty, cfg.top_transformer.penalty_weight, cfg.top_transformer.anneal)

    if pathlib.Path(cfg.top_transformer.load_path).is_file():
        print(f"Loading Flow4Flow from model: {cfg.top_transformer.load_path}")
        f4flow.load_state_dict(torch.load(cfg.top_transformer.load_path, map_location=device))
    else:
        print("Training Flow4Flow model")
        train_f4f(f4flow, base_data, val_base_data,
                  cfg.top_transformer.nepochs, cfg.top_transformer.lr, cfg.general.ncond,
                  outputpath, name='f4f', device=device, gclip=cfg.top_transformer.gclip)

    with torch.no_grad():
        f4flow.to(device)
        test_data = get_data(n_points)

        test_points = test_data.get_default_eval(6)
        for con in test_points:
            # Handle the broadcasting
            left_data, left_cond, right_cond = [d.to(device) \
                                                for d in ConditionalDataToTarget(test_data.get_tuple(), con).paired()]
            # Transform the data
            transformed, _ = f4flow.batch_transform(left_data, left_cond, right_cond, batch_size=1000)
            # Plot the output densities
            plot_data(transformed, outputpath / f'flow_for_flow_{tensor_to_str(con)}.png')
            # Get the transformation that results from going via the base distributions
            left_bd_enc = f4flow.base_flow_left.transform_to_noise(left_data, left_cond)
            right_bd_dec, _ = f4flow.base_flow_right._transform.inverse(left_bd_enc, right_cond)
            # Plot how each point is shifted
            plot_arrays({
                'Input Data': left_data,
                'FFF': transformed,
                'BdTransfer': right_bd_dec
            }, outputpath, f'{con.item():.2f}')

            ##dump data
            df = dump_to_df(left_data, left_cond, right_cond, transformed, left_bd_enc, right_bd_dec,
                            col_names=['input_x','input_y','left_cond','right_cond',
                                    'transformed_x','transformed_y','left_enc_x','left_enc_y',
                                    'base_transfer_x','base_transfer_y'])
            df.to_hdf(outputpath / 'eval_data.h5', f'f4f_{con:.2f}'.replace('.','_'))


if __name__ == "__main__":
    main()
