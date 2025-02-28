import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

project_root = os.path.abspath("..")  # If notebooks is one folder above src
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Get the parent directory (one level above)
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.insert(0,parent_dir + "/socmatching")

import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch_optimizer as torch_optim

from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
from omegaconf import OmegaConf
import yaml
import time

from src.soc.model import FullyConnectedUNet
from src.soc.settings import get_energy, get_Rfunc
from src.soc.loss import compute_loss, compute_l2_error
from src.soc.utils import exact_eigfunctions

import wandb

from socmatching.SOC_matching.utils import (
    get_folder_name,
    get_file_name,
    control_objective,
    save_results,
    compute_EMA,
    normalization_constant,
    stochastic_trajectories
)

from socmatching.SOC_matching.experiment_settings.settings import define_variables

def setup(rank, world_size):
    """Initialize process group for distributed training."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Destroy process group to free resources."""
    dist.destroy_process_group()

def main(rank, world_size):

    setup(rank, world_size)

    # Set device for the current process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.init()
    torch.cuda.set_device(device)
    base_cfg = OmegaConf.load('socm_config.yaml')
    efc_cfg = OmegaConf.load('efc_config.yaml')
    cfg = OmegaConf.merge(base_cfg, efc_cfg)

    if rank == 0:
        wandb.init(
            project="neural-eigenfunction-learner",
            group = 'OU-QHARD_d40',
            name = f'EFC (0-0.75T)',
            config=dict(cfg),
            job_type='eval'
        )
    else:
        wandb.init(mode="disabled")

    torch.manual_seed(cfg.method.seed)
    cfg.method.device = "cuda"
    ts = torch.linspace(0, cfg.method.T, cfg.method.num_steps + 1).to(cfg.method.device)
    
    x0, sigma, optimal_sde, neural_sde, _ = define_variables(cfg, ts)
    if optimal_sde is not None:
        ground_truth_control = optimal_sde.u
    else:
        ground_truth_control = None

    compute_L2_error = ground_truth_control is not None
    compute_eigf_error = cfg.method.setting[:12] == 'OU_quadratic'

    state0 = x0.repeat(cfg.optim.batch_size, 1)

    model = FullyConnectedUNet(dim=cfg.method.d,
                                k = 1,
                                hdims = cfg.arch.hdims).to(cfg.method.device)
    
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(params=ddp_model.parameters(),
                           lr=cfg.optim.adam_lr, 
                           eps=cfg.optim.adam_eps)
    
    warmup_epochs = cfg.optim.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda epoch: ((epoch + 1) / warmup_epochs) if epoch < warmup_epochs else
                                  0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (cfg.method.num_iterations - warmup_epochs)))
    )
    
    energy = get_energy(cfg, neural_sde)
    Rfunc = get_Rfunc(cfg, neural_sde)

    #state0 = energy.exact_sample((cfg.optim.batch_size,))

    neural_sde.use_learned_control = False
    neural_sde.u = ground_truth_control

    ###### Train eigenfunction model ######
    with torch.inference_mode(False):
        with torch.enable_grad():
            for itr in range(cfg.method.num_iterations):
                ddp_model.train()
                # Sample batch from distributed sampler
                torch.manual_seed(world_size * itr)

                # generate samples
                x = energy.exact_sample((cfg.method.n_samples_loss,)).to(cfg.method.device)
                x.requires_grad_()
                
                Rx = Rfunc(x)

                loss, var_loss, orth_loss = compute_loss(ddp_model, x, Rx, cfg.loss.beta)

                loss.backward()

                with torch.no_grad():
                    optimizer.step()
                    optimizer.zero_grad()

                wandb.log({'loss': loss,
                           'var_loss': var_loss,
                           'orth_loss': orth_loss})
                
                ddp_model.eval()

                if itr % 100 == 0 and rank == 0 and compute_L2_error:
                    # compute trajectories for L2 evaluation
                    (
                    states,
                    _,
                    _,
                    _,
                    log_path_weight_deterministic,
                    log_path_weight_stochastic,
                    log_terminal_weight,
                    _,
                    ) = stochastic_trajectories(
                        neural_sde,
                        state0,
                        ts.to(state0),
                        cfg.method.lmbd,
                        detach=False)
                    
                    states.requires_grad_(True)
                    
                    # (T,N,d)
                    target_control = ground_truth_control(ts, states, t_is_tensor=True).detach().cpu()

                    eval_idx1 = int(( 1 - cfg.method.eval_frac) / 2 * len(ts))
                    eval_idx2 = int((1 + cfg.method.eval_frac) / 2 * len(ts))
                    states = states[:eval_idx2,:,:]
                    target_control = target_control[:eval_idx2,:,:]

                    stacked_states = states.reshape((states.shape[0]*states.shape[1],states.shape[2]))
                    assert stacked_states[1,2] == states[0,1,2]

                    if cfg.method.k == 1:
                        fx = ddp_model(stacked_states)
                        grad_fx = torch.autograd.grad(outputs = fx,
                            inputs = stacked_states, 
                            grad_outputs=torch.ones_like(fx),
                            retain_graph=True,
                            create_graph=True)[0].detach().cpu()
                        
                        fx = fx.detach().cpu()
                        fx = (torch.abs(fx) + 1e-2) * torch.sign(fx)
                        learned_control = cfg.method.lmbd * grad_fx / fx 
                        
                        reshaped_learned_control = learned_control.reshape(target_control.shape)

                    else:
                        raise NotImplementedError

                    norm_sqd_diff = compute_l2_error(target_control, reshaped_learned_control)
                    print(f"{itr}: l2 error {norm_sqd_diff.detach():5.6E}")
                
                    wandb.log({'l2_error': norm_sqd_diff.detach()})

                    # if compute_eigf_error:
                    #     print(states.shape)
                    #     print(states.max())
                    #     norm = torch.mean(ddp_model(x)[:,0]**2).sqrt().detach().cpu()
                    #     exact_fx = exact_eigfunctions(stacked_states.detach(), 1, neural_sde, cfg, return_grad=False).cpu()
                    #     eigf_sq_diff = torch.mean((exact_fx-fx/norm)**2)
                    #     neg_eigf_sq_diff = torch.mean((exact_fx+fx/norm)**2)
                    #     exact_sq_sum = torch.mean(exact_fx**2)

                    #     l2_err = min(eigf_sq_diff,neg_eigf_sq_diff) / exact_sq_sum
                    #     wandb.log({'eigf_l2_error': l2_err})
                    #     print(f"{itr}: eigf l2 error {l2_err:5.6E}")

                # if itr % 10 == 0 and rank == 0 and compute_eigf_error:
                #     fx = ddp_model(x.detach()).cpu()
                #     norm = torch.mean(fx[:,0]**2).sqrt()
                #     exact_fx = exact_eigfunctions(x.detach(), 1, neural_sde, cfg, return_grad=False).cpu()
                #     eigf_sq_diff = torch.mean((exact_fx-fx/norm)**2)
                #     neg_eigf_sq_diff = torch.mean((exact_fx+fx/norm)**2) 

                #     wandb.log({'eigf_l2_error': min(eigf_sq_diff,neg_eigf_sq_diff)})
                #     print(f"{itr}: eigf l2 error {min(eigf_sq_diff,neg_eigf_sq_diff):5.6E}")
                
                scheduler.step()

        cleanup()  # Cleanup process group after training
        if rank == 0:
            wandb.finish()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)