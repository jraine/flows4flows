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
    return train(*args, **kwargs)

def train_f4f_forward(*args, **kwargs):
    return train(*args, **kwargs, rand_perm_target=True, inverse=False)

def train_f4f_inverse(*args, **kwargs):
    return train(*args, **kwargs, rand_perm_target=True, inverse=True)


@hydra.main(version_base=None, config_path="conf/", config_name="nocond_default")
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
    base_data_l,val_base_data_l = None,None
    base_data_r,val_base_data_r = None,None
    # base_data_l = get_data(cfg.base_dist.data, int(1e4), batch_size=cfg.base_dist.batch_size)
    # val_base_data_l = get_data(cfg.base_dist.data, int(1e4), batch_size=1000)
    cfg.general.ncond = None if cfg.general.ncond == 0 else cfg.general.ncond

    # Train base1
    base_flow_l,base_flow_r = None,None
    for label,base_data,val_base_data,bd_conf,base_flow in zip(['left','right'],
                                       [base_data_l, base_data_r],
                                       [val_base_data_l, val_base_data_r],
                                       [cfg.base_dist.left,cfg.base_dist.right],
                                       [base_flow_l,base_flow_r]):

        base_data = get_data(bd_conf.data, int(1e4), batch_size=bd_conf.batch_size)
        val_base_data = get_data(bd_conf.data, int(1e4), batch_size=1000)


        base_flow = BaseFlow(spline_inn(cfg.general.data_dim,
                                        nodes=bd_conf.nnodes,
                                        num_blocks=bd_conf.nblocks,
                                        num_stack=bd_conf.nstack,
                                        activation=get_activation(bd_conf.activation),
                                        num_bins=bd_conf.nbins, 
                                        context_features=cfg.general.ncond
                                    ),
                             StandardNormal([cfg.general.data_dim])
                            )
        
        
        if pathlib.Path(bd_conf.load_path).is_file():
            print(f"Loading base_{label} from model: {bd_conf.load_path}")
            base_flow.load_state_dict(torch.load(bd_conf.load_path))
        else:
            print(f"Training base_{label} distribution")                        
            train_base(base_flow, base_data, val_base_data,
                    bd_conf.nepochs, bd_conf.lr, cfg.general.ncond,
                    outputpath, name=f'base_{label}', device=device)

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
                                         base_flow_l,
                                         base_flow_r)
     
    
    print("Training Flow4Flow model forwards")
    train_f4f_forward(f4flow, base_data, val_base_data,
                      cfg.top_transformer.nepochs, cfg.top_transformer.lr, cfg.general.data_dim,
                      outputpath, name='f4f_fwd', device=device)
    
    print("Training Flow4Flow model backwards")
    train_f4f_inverse(f4flow, base_data, val_base_data,
                      cfg.top_transformer.nepochs, cfg.top_transformer.lr, cfg.general.data_dim,
                      outputpath, name='f4f_inv', device=device)


if __name__ == "__main__":
    main()