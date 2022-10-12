import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib
import ffflows
from ffflows.models import BaseFlow
from ffflows.utils import set_trainable

import torch

from nflows import transforms
from nflows.distributions import StandardNormal
from nflows.flows import Flow

from utils import get_activation, get_data, get_flow4flow, train, spline_inn
from plot import plot_training, plot_data, plot_arrays

def train_base(*args, **kwargs):
    return train(*args, **kwargs, rand_perm_target=False)

def train_f4f(*args, **kwargs):
    return train(*args, **kwargs, rand_perm_target=True)


@hydra.main(version_base=None, config_path="conf/", config_name="cond_jointbase_default")
def main(cfg : DictConfig) -> None:

    print("Configuring job with following options")
    print(OmegaConf.to_yaml(cfg))
    outputpath=pathlib.Path(cfg.output.save_dir+'/'+cfg.output.name)
    outputpath.mkdir(parents=True,exist_ok=True)
    with open(outputpath / f"{cfg.output.name}.yaml", 'w') as file:
        OmegaConf.save(config=cfg, f=file)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get training data
    base_data = get_data(cfg.base_dist.data, int(1e4), batch_size=cfg.base_dist.batch_size)
    val_base_data = get_data(cfg.base_dist.data, int(1e4), batch_size=1000)

    # Train base
    base_flow = BaseFlow(spline_inn(cfg.general.data_dim,
                                nodes=cfg.base_dist.nnodes,
                                num_blocks=cfg.base_dist.nblocks,
                                num_stack=cfg.base_dist.nstack,
                                activation=get_activation(cfg.base_dist.activation),
                                num_bins=cfg.base_dist.nbins, 
                                context_features=cfg.general.ncond
                                ),
                        StandardNormal([cfg.general.data_dim])
                    )
    if pathlib.Path(cfg.base_dist.load_path).is_file():
        print(f"Loading base from model: {cfg.base_dist.load_path}")
        base_flow.load_state_dict(torch.load(cfg.base_dist.load_path))
    else:
        print("Training base distribution")                        
        train_base(base_flow, base_data, val_base_data,
                cfg.base_dist.nepochs, cfg.base_dist.lr, cfg.general.ncond,
                outputpath, name='base', device=device)

    set_trainable(base_flow,False)

    # Train Flow4Flow
    f4flow = get_flow4flow(cfg.top_transformer.flow4flow,
                                         spline_inn(cfg.general.data_dim,
                                                    nodes=cfg.top_transformer.nnodes,
                                                    num_blocks=cfg.top_transformer.nblocks,
                                                    num_stack=cfg.top_transformer.nstack,
                                                    activation=get_activation(cfg.top_transformer.activation),
                                                    num_bins=cfg.top_transformer.nbins, 
                                                    context_features=cfg.general.ncond
                                                   ),
                                         base_flow)
     
    
    print("Training Flow4Flow model")
    train_f4f(f4flow, base_data, val_base_data,
               cfg.base_dist.nepochs, cfg.base_dist.lr, cfg.general.data_dim,
               outputpath, name='f4f', device=device)


if __name__ == "__main__":
    main()