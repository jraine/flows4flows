import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib
from ffflows.models import BaseFlow
from ffflows.utils import set_trainable

import torch
from torch.utils.data import DataLoader

from nflows.distributions import StandardNormal

from ffflows.utils import get_activation, get_flow4flow, train, spline_inn, set_penalty
import matplotlib.pyplot as plt

from ffflows.data.dist_to_dist import PairedConditionalDataToTarget
# CHANGED: new Conditional dataset
from ffflows.data.conditional_plane import ConditionalDataset

import numpy as np

np.random.seed(42)
torch.manual_seed(42)


def train_base(*args, **kwargs):
    return train(*args, **kwargs)


# CHANGED: rand_perm_target from True -> False
def train_f4f_forward(*args, **kwargs):
    return train(*args, **kwargs, rand_perm_target=False, inverse=False)


def train_f4f_inverse(*args, **kwargs):
    return train(*args, **kwargs, rand_perm_target=False, inverse=True)

@hydra.main(version_base=None, config_path="conf/", config_name="cond_transfer")
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get training data
    # CHANGED: LHCO data
    train_sim_data = torch.from_numpy(np.load("LHCO_data/train_sim_data.npy")).to(torch.float32)
    val_sim_data = torch.from_numpy(np.load("LHCO_data/val_sim_data.npy")).to(torch.float32)
    train_dat_data = torch.from_numpy(np.load("LHCO_data/train_dat_data.npy")).to(torch.float32)
    val_dat_data = torch.from_numpy(np.load("LHCO_data/val_dat_data.npy")).to(torch.float32)

    train_sim_cont = torch.from_numpy(np.load("LHCO_data/train_sim_cont.npy").reshape(-1, 1)).to(torch.float32)
    val_sim_cont = torch.from_numpy(np.load("LHCO_data/val_sim_cont.npy").reshape(-1, 1)).to(torch.float32)
    train_dat_cont = torch.from_numpy(np.load("LHCO_data/train_dat_cont.npy").reshape(-1, 1)).to(torch.float32)
    val_dat_cont = torch.from_numpy(np.load("LHCO_data/val_dat_cont.npy").reshape(-1, 1)).to(torch.float32)
    
    train_left_data = DataLoader(dataset=ConditionalDataset(train_sim_data, train_sim_cont),
        batch_size=cfg.base_dist.batch_size,shuffle=True)
    
    val_left_data = DataLoader(dataset=ConditionalDataset(val_sim_data, val_sim_cont), batch_size=1000, shuffle = False)
    
    train_right_data = DataLoader(dataset=ConditionalDataset(train_dat_data, train_dat_cont),
        batch_size=cfg.top_transformer.batch_size,shuffle=True)
    
    val_right_data = DataLoader(dataset=ConditionalDataset(val_dat_data, val_dat_cont), batch_size=1000, shuffle = False)
    
    ncond_base = None if cfg.general.ncond == 0 else cfg.general.ncond

    # Train base
    base_flow_l, base_flow_r = [BaseFlow(spline_inn(cfg.general.data_dim,
                                                    nodes=cfg.base_dist.left.nnodes,
                                                    num_blocks=cfg.base_dist.left.nblocks,
                                                    num_stack=cfg.base_dist.left.nstack,
                                                    tail_bound=4.0,
                                                    activation=get_activation(cfg.base_dist.left.activation),
                                                    num_bins=cfg.base_dist.left.nbins,
                                                    context_features=ncond_base
                                                    ),
                                         StandardNormal([cfg.general.data_dim]))
                                     for bd_conf in [cfg.base_dist.left, cfg.base_dist.right]]
                                       
                        
    for label, base_data, val_base_data, bd_conf, base_flow in zip(['left', 'right'],
                                                                   [train_left_data, train_right_data],
                                                                   [val_left_data, val_right_data],
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
                                      flow_for_flow=True
                                      ),
                           distribution_right=base_flow_r,
                           distribution_left=base_flow_l)

    set_penalty(f4flow, cfg.top_transformer.penalty, cfg.top_transformer.penalty_weight, cfg.top_transformer.anneal)
    direction = cfg.top_transformer.direction.lower()
    if pathlib.Path(cfg.top_transformer.load_path).is_file():
        print(f"Loading Flow4Flow from model: {cfg.top_transformer.load_path}")
        f4flow.load_state_dict(torch.load(cfg.top_transformer.load_path, map_location=device))

    else:
    # CHANGED: dataset                                                              
        if (direction == 'forward' or direction == 'both'):
            print("Training Flow4Flow model forwards")
            train_f4f_forward(f4flow,
                              train_left_data,
                              val_left_data, 
                              cfg.top_transformer.nepochs, cfg.top_transformer.lr, ncond_base,
                              outputpath, name='f4f_fwd', device=device, gclip=cfg.top_transformer.gclip)

        if (direction == 'inverse' or direction == 'both'):
            print("Training Flow4Flow model backwards")
            train_f4f_inverse(f4flow,
                              train_right_data,
                              val_right_data, 
                              cfg.top_transformer.nepochs, cfg.top_transformer.lr, ncond_base,
                              outputpath, name='f4f_inv', device=device, gclip=cfg.top_transformer.gclip)
                              

  
if __name__ == "__main__":
    main()
