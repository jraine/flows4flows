import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib
import ffflows

import torch
from torch.utils.data import DataLoader

from nflows import transforms
from nflows.distributions import StandardNormal
from nflows.flows import Flow

from utils import get_data, get_flow4flow

def train(model, train_data, val_data, n_epochs, learning_rate, ncond, path, name, rand_perm_target, device='cpu'):
    
    save_path = pathlib.Path(path / name)
    save_path.mkdir(parents=True,exist_ok=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_steps = len(train_data) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, last_epoch=-1,
                                                           eta_min=0)
    train_loss = torch.zeros(n_epochs)
    valid_loss = torch.zeros(n_epochs)
    for epoch in range(n_epochs):
        t_loss = []

        for step, data in enumerate(train_data):
            model.train()

            optimizer.zero_grad()
            if ncond is not None:
                inputs, context_l = data
                context_r = torch.rand_perm(context_l) if rand_perm_target else None
            else:
                inputs, context_l, context_r = data, None, None

            noise, logprob = model.transform(inputs, context_l=context_l, context_r=context_r)

            logprob.backward()
            optimizer.step()
            scheduler.step()
            t_loss.append(logprob.item())
        
        train_loss[epoch] = torch.tensor(t_loss).mean()
        v_loss = torch.zeros(len(val_data))
        for v_step, data in enumerate(val_data):
            if ncond is not None:
                inputs, context_l = data
                context_r = torch.rand_perm(context_l) if rand_perm_target else None
            else:
                inputs, context_l, context_r = data, None, None

            with torch.no_grad():
                v_loss[v_step] = model.log_prob(inputs, context_l=context_l, context_r=context_r)
        valid_loss[epoch] = v_loss.mean()

        torch.save(model.state_dict(), save_path / f'epoch_{epoch}_valloss_{valid_loss[epoch]}')

    ###insert saving of losses and plots stuff
    return

def train_base(*args, **kwargs):
    train(*args, **kwargs, rand_perm_target=False)

def train_f4f(*args, **kwargs):
    train(*args, **kwargs, rand_perm_target=True)


@hydra.main(version_base=None, config_path="conf", config_name="cond_jointbase_default.conf")
def main(cfg : DictConfig) -> None:

    print("Configuring job with following options")
    print(OmegaConf.to_yaml(cfg))
    outputpath=pathlib.Path(cfg.output.save_dir+'/'+cfg.output.name)
    outputpath.mkdir(parents=True,exist_ok=True)
    with open(outputpath+f'{cfg.output.name}') as file:
        OmegaConf.save(config=OmegaConf.to_yaml(cfg), f=file)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #train base
    ## get base train dataset
    base_data = DataLoader(dataset=get_data(cfg.base_dist.data), 
                           batch_size=cfg.base_dist.batch_size)
    val_base_data = DataLoader(dataset=get_data(cfg.base_dist.data), 
                               batch_size=1_000)

    if pathlib.Path(cfg.base_dist.load_path).is_file:
        base_flow = torch.load(cfg.base_dist.load_path)
    else:
        base_flow = Flow(ffflows.utils.spline_inn(cfg.general.data_dim,
                                                  nodes=cfg.base_dist.nnodes,
                                                  num_blocks=cfg.base_dist.nblocks,
                                                  num_stack=cfg.base_dist.nstack,
                                                  activation=cfg.base_dist.activation,
                                                  num_bins=cfg.general.nbins, 
                                                  context_features=cfg.general.ncond
                                                 ),
                         StandardNormal([cfg.general.data_dim])
                        )
    train_base(base_flow, base_data, val_base_data,
               cfg.base_dist.nepochs, cfg.base_dist.lr, cfg.general.data_dim,
               outputpath, name='base', device=device)


    ffflows.utils.set_trainable(base_flow,False)

    f4flow = get_flow4flow(cfg.top_transformer.flow4flow,
                                         ffflows.utils.spline_inn(cfg.general.data_dim,
                                                                 nodes=cfg.base_dist.nnodes,
                                                                 num_blocks=cfg.base_dist.nblocks,
                                                                 num_stack=cfg.base_dist.nstack,
                                                                 activation=cfg.base_dist.activation,
                                                                 num_bins=cfg.general.nbins, 
                                                                 context_features=cfg.general.ncond
                                                                 ),
                                         base_flow)
     
    
    train_f4f(f4flow, base_data, val_base_data,
               cfg.base_dist.nepochs, cfg.base_dist.lr, cfg.general.data_dim,
               outputpath, name='f4f', device=device)


if __name__ == "__main__":
    main()