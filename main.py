import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

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
import pickle

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

from socmatching.SOC_matching.method import (
    SOC_Solver,
)

from src.soc.settings import get_energy, get_Rfunc
from src.soc.utils import exact_eigfunctions

from src.soc.method import EigenSolver

from src.experiment_settings.socm_settings.settings import define_variables as define_variables_socm
from src.experiment_settings.efc_settings.settings import define_variables as define_variables_efc

# Code essentially copied from SOC_matching main.py

def setup(rank, world_size):
    """Initialize process group for distributed training."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Destroy process group to free resources."""
    dist.destroy_process_group()

def main(rank, world_size, algorithms):
    """
    Config entries expected:
        method.seed: random seed
        method.algorithm: algorithm name
        method.setting: problem setting
        method.T: final T
        method.num_stemps: number of timesteps
        optim.batch_size: batch size for optimization
        optim.M_lr: learning rate
        optim.nabla_V_lr: learning rate
    """
    setup(rank, world_size)

    # Set device for the current process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.init()
    torch.cuda.set_device(device)
    algorithm = algorithms[rank]

    experiment_cfg = OmegaConf.load('experiment_config.yaml')
    if algorithm == "EFC":
        method_cfg = OmegaConf.load('efc_config.yaml')
    else:
        method_cfg = OmegaConf.load('socm_config.yaml')

    cfg = OmegaConf.merge(experiment_cfg, method_cfg)

    cfg.method.algorithm = algorithm

    wandb.init(
        project="neural-eigenfunction-learner",
        group = cfg.experiment.name,
        name = algorithm,
        config=dict(cfg),
        job_type='eval'
    )

    torch.manual_seed(cfg.method.seed)
    cfg.method.device = "cuda"

    appendix = ""
    if algorithm == "EFC":
        appendix = f'_{cfg.method.k}'
        if cfg.arch.prior is not None:
            appendix += f'_{cfg.arch.prior}'

    experiment_path = f'experiments/{cfg.experiment.name}/{algorithm}' + appendix

    os.makedirs(experiment_path,exist_ok=True)
    OmegaConf.save(cfg, experiment_path + '/cfg.yaml')

    if algorithm != "EFC":
        cfg.method.T = cfg.method.train_T
        cfg.method.num_steps = cfg.method.train_steps

    ts = torch.linspace(0, cfg.method.T, cfg.method.num_steps + 1).to(cfg.method.device)
    
    if algorithm == "EFC":
        x0, sigma, optimal_sde, neural_sde = define_variables_efc(cfg, ts)
        normalization_const = 0
    else:
        x0, sigma, optimal_sde, neural_sde, u_warm_start = define_variables_socm(cfg, ts)
    
    if optimal_sde is not None:
        ground_truth_control = optimal_sde.u
    else:
        ground_truth_control = None

    state0 = x0.repeat(cfg.optim.batch_size, 1)

    if algorithm != "EFC" and rank == 0:
        ########### Compute normalization constant and control L2 error for initial control ############
        print(
            f"Estimating normalization constant and control L2 error for initial control..."
        )
        (
            normalization_const,
            normalization_const_std_error,
            norm_sqd_diff_mean,
        ) = normalization_constant(
            neural_sde,
            state0,
            ts,
            cfg,
            n_batches_normalization=512,
            ground_truth_control=ground_truth_control,
        )
        print(
            f"Normalization_constant (mean and std. error): {normalization_const:5.8E} {normalization_const_std_error:5.8E}"
        )
        if ground_truth_control is not None:
            print(
                f"Control L2 error for initial control: {norm_sqd_diff_mean / normalization_const}"
            )

    normalization_const = torch.tensor(normalization_const if rank == 0 else 0.0, device=device)
    dist.broadcast(normalization_const, src=0)

    ####### Initialize Algorithm ########
    if algorithm == "EFC":
        solver = EigenSolver(
            neural_sde,
            x0,
            ground_truth_control,
            T=cfg.method.T,
            num_steps=cfg.method.num_steps,
            lmbd=cfg.method.lmbd,
            d=cfg.method.d,
            sigma=sigma,
            langevin_burnin_steps=cfg.method.langevin_burnin_steps,
            langevin_sample_steps=cfg.method.langevin_sample_steps,
            langevin_dt=cfg.method.langevin_dt
        )
    else:
        solver = SOC_Solver(
            neural_sde,
            x0,
            ground_truth_control,
            T=cfg.method.T,
            num_steps=cfg.method.num_steps,
            lmbd=cfg.method.lmbd,
            d=cfg.method.d,
            sigma=sigma,
        )

    if algorithm == "SOCM_exp":
        solver.gamma = torch.nn.Parameter(
            torch.tensor([cfg.method.gamma]).to(cfg.method.device)
        )
    elif algorithm != "EFC":
        solver.gamma = cfg.method.gamma


    ####### Set optimizer ########
    scheduler = None

    if algorithm == "EFC":
        warmup_epochs = cfg.optim.warmup_epochs

        if cfg.arch.joint:
            
            optimizer = optim.Adam(params=solver.neural_sde.eigf_model.parameters(),
                            lr=cfg.optim.adam_lr, 
                            eps=cfg.optim.adam_eps)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda = lambda epoch: ((epoch + 1) / warmup_epochs) if epoch < warmup_epochs else
                                    0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (cfg.method.num_iterations - warmup_epochs)))
            )
        else:
            optimizers = [optim.Adam(params=solver.neural_sde.eigf_models[i].parameters(),
                            lr=cfg.optim.adam_lr, 
                            eps=cfg.optim.adam_eps) for i in range(cfg.method.k)]
            schedulers = [torch.optim.lr_scheduler.LambdaLR(
            optimizers[i],
            lr_lambda = lambda epoch: ((epoch + 1) / warmup_epochs) if epoch < warmup_epochs else
                                    0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (cfg.method.num_iterations - warmup_epochs)))
            ) for i in range(cfg.method.k)]
        
    elif algorithm == "moment":
        optimizer = torch.optim.Adam(
            [{"params": solver.neural_sde.parameters()}]
            + [{"params": solver.y0, "lr": cfg.optim.y0_lr}],
            lr=cfg.optim.nabla_V_lr,
            eps=cfg.optim.adam_eps,
        )
    elif algorithm == "SOCM_exp":
        optimizer = torch.optim.Adam(
            [{"params": solver.neural_sde.parameters()}]
            + [{"params": solver.gamma, "lr": cfg.optim.M_lr}],
            lr=cfg.optim.nabla_V_lr,
            eps=cfg.optim.adam_eps,
        )
    elif algorithm == "SOCM":
        if cfg.method.use_stopping_time:
            optimizer = torch.optim.Adam(
                [{"params": solver.neural_sde.nabla_V.parameters()}]
                + [
                    {
                        "params": solver.neural_sde.M.sigmoid_layers.parameters(),
                        "lr": cfg.optim.M_lr,
                    }
                ]
                + [
                    {
                        "params": solver.neural_sde.gamma,
                        "lr": cfg.optim.M_lr,
                    }
                ]
                + [
                    {
                        "params": solver.neural_sde.gamma2,
                        "lr": cfg.optim.M_lr,
                    }
                ],
                lr=cfg.optim.nabla_V_lr,
                eps=cfg.optim.adam_eps,
            )
        else:
            optimizer = torch.optim.Adam(
                [{"params": solver.neural_sde.nabla_V.parameters()}]
                + [
                    {
                        "params": solver.neural_sde.M.sigmoid_layers.parameters(),
                        "lr": cfg.optim.M_lr,
                    }
                ]
                + [
                    {
                        "params": solver.neural_sde.gamma,
                        "lr": cfg.optim.M_lr,
                    }
                ],
                lr=cfg.optim.nabla_V_lr,
                eps=cfg.optim.adam_eps,
            )
    elif algorithm == "rel_entropy":
        optimizer = torch.optim.Adam(
            solver.parameters(), lr=cfg.optim.nabla_V_lr, eps=cfg.optim.adam_eps
        )
    else:
        optimizer = torch.optim.Adam(
            solver.parameters(), lr=cfg.optim.nabla_V_lr, eps=cfg.optim.adam_eps
        )

    if algorithm == "EFC" and not cfg.arch.joint:
        for optimizer in optimizers:
            optimizer.zero_grad()
    else:
        optimizer.zero_grad()

    solver.algorithm = cfg.method.algorithm
    solver.training_info = dict()
    training_info = solver.training_info
    training_variables = [
        "time_per_iteration",
        "loss",
        "norm_sqd_diff",
        "norm_sqd_diff_single"
    ]
    control_objective_variables = [
        "control_objective_mean",
        "control_objective_std_err",
        "control_objective_itr",
        "trajectories",
    ]
    for var in training_variables:
        training_info[var] = []
    for var in control_objective_variables:
        training_info[var] = []

    compute_L2_error = ground_truth_control is not None
    EMA_weight_mean_coeff = 0.002

    #energy = get_energy(cfg, neural_sde)
    wandb_log = {}

    ###### Train control ######
    with torch.inference_mode(False):
        with torch.enable_grad():
            for itr in range(cfg.method.num_iterations):
                solver.train()
                torch.manual_seed(itr)
                start = time.time()

                compute_control_objective = (
                    itr == 0
                    or itr % cfg.method.compute_control_objective_every
                    == cfg.method.compute_control_objective_every - 1
                    or itr == cfg.method.num_iterations - 1
                )
                verbose = itr == 0

                if algorithm == "EFC":
                    (
                    loss,
                    var_loss,
                    orth_loss
                    ) = solver.loss(
                        sample_size=cfg.method.n_samples_loss,
                        verbose=True,
                        beta=cfg.loss.beta,
                    )
                    if cfg.arch.joint:
                        wandb_log['loss'] = loss.detach()
                        wandb_log['var_loss'] = var_loss.detach()
                        wandb_log['orth_loss'] = orth_loss.detach()
                    else:
                        for i in range(cfg.method.k):
                            wandb_log[f'loss_{i}'] = loss[i].detach()
                            wandb_log[f'var_loss_{i}'] = var_loss[i].detach()
                            wandb_log[f'orth_loss_{i}'] = orth_loss[i].detach()
                else:
                    (
                    loss,
                    _,
                    control_objective_mean,
                    control_objective_std_err,
                    trajectory,
                    weight_mean,
                    weight_std,
                    stop_indicators,
                    ) = solver.loss(
                        cfg.optim.batch_size,
                        compute_L2_error=False,
                        algorithm=algorithm,
                        optimal_control=ground_truth_control,
                        compute_control_objective=compute_control_objective,
                        total_n_samples=cfg.method.n_samples_control,
                        verbose=verbose,
                        u_warm_start=u_warm_start,
                        use_warm_start=cfg.method.use_warm_start,
                        use_stopping_time=cfg.method.use_stopping_time,
                    )

                    if (
                        algorithm == "SOCM_const_M"
                        or algorithm == "SOCM_exp"
                        or algorithm == "SOCM"
                        or algorithm == "SOCM_adjoint"
                        or algorithm == "cross_entropy"
                    ):
                        loss = loss / normalization_const
                    elif algorithm == "variance":
                        loss = loss / normalization_const**2
                    
                    wandb_log['loss'] = loss.detach()
                
                # Backward pass
                if algorithm != "EFC" or cfg.arch.joint:
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                else:
                    for i in range(loss.shape[0]):
                        # backward pass on i-th loss function
                        grads = torch.autograd.grad(loss[i], solver.neural_sde.eigf_models[i].parameters(), retain_graph=True)

                        for param, grad in zip(solver.neural_sde.eigf_models[i].parameters(),grads):
                            param.grad = grad

                        if i == 0 and neural_sde.prior == "positive":
                            torch.nn.utils.clip_grad_norm_(solver.neural_sde.eigf_models[i].parameters(), 1.0)
                        
                        optimizers[i].step()
                        optimizers[i].zero_grad()
                        
                        if schedulers[i] is not None:
                            schedulers[i].step()
                
                with torch.no_grad():
                    end = time.time()
                    time_per_iteration = end - start

                    if algorithm != "EFC":
                        normalization_const = compute_EMA(
                            weight_mean.detach(),
                            normalization_const,
                            EMA_coeff=EMA_weight_mean_coeff,
                            itr=itr,
                        )

                    training_info["time_per_iteration"].append(time_per_iteration)
                    training_info["loss"].append(loss.detach())
                    
                    if itr % 100 == 0 and compute_L2_error:
                        solver.eval()

                        if algorithm == "EFC":
                            solver.compute_eigvals(beta=cfg.loss.beta)
                            
                            solver.neural_sde.compute_inner_prods(samples = solver.samples)

                        # use trajectories from ground truth control
                        neural_sde.use_learned_control = False
                        neural_sde.u = ground_truth_control
                        (
                        states,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        ) = stochastic_trajectories(
                            neural_sde,
                            state0,
                            ts.to(state0),
                            cfg.method.lmbd,
                            detach=True)
                        
                        neural_sde.use_learned_control = True
                        
                        target_control = ground_truth_control(ts, states, t_is_tensor=True)
                        
                        # only evaluate in [0, eval_frac * T]
                        eval_idx = int((cfg.method.eval_frac) * len(ts))
                        states = states[:eval_idx,:,:]
                        target_control = target_control[:eval_idx,:,:]

                        learned_control = neural_sde.control(ts[:eval_idx],states)
                        norm_sqd_diff = torch.sum(
                            (target_control - learned_control) ** 2
                            / (target_control.shape[0] * target_control.shape[1])
                        )
                        print(f"{itr}: l2 error {norm_sqd_diff.detach():5.6E}")
                        
                        norm_sqd_diff_single = None
                        if algorithm == "EFC" and cfg.method.k > 1:
                            learned_control = neural_sde.control(ts[:eval_idx],states,1)
                            norm_sqd_diff_single = torch.sum(
                                (target_control - learned_control) ** 2
                                / (target_control.shape[0] * target_control.shape[1])
                            )
                            print(f"{itr}: l2 error {norm_sqd_diff.detach():5.6E}")
                                                    
                        # x = solver.samples

                        # exact_fx = exact_eigfunctions(x, 2, neural_sde, cfg, return_grad=False).cpu()

                        # fx = solver.neural_sde.eigf_models[0](x).detach().cpu().squeeze(1)
                        # norm = torch.mean(fx**2).sqrt().detach().cpu()
                        
                        # eigf_sq_diff = torch.mean((exact_fx[:,0]-fx/norm)**2)
                        # neg_eigf_sq_diff = torch.mean((exact_fx[:,0]+fx/norm)**2)
                        # exact_sq_sum = torch.mean(exact_fx[:,0]**2)

                        # l2_err = min(eigf_sq_diff,neg_eigf_sq_diff) / exact_sq_sum
                        # wandb_log['eigf_l2_error_0'] = l2_err
                        
                        # fx = solver.neural_sde.eigf_models[1](x).detach().cpu().squeeze(1)
                        # norm = torch.mean(fx**2).sqrt().detach().cpu()
                        
                        # eigf_sq_diff = torch.mean((exact_fx[:,1]-fx/norm)**2)
                        # neg_eigf_sq_diff = torch.mean((exact_fx[:,1]+fx/norm)**2)
                        # exact_sq_sum = torch.mean(exact_fx[:,1]**2)

                        # l2_err = min(eigf_sq_diff,neg_eigf_sq_diff) / exact_sq_sum
                        # wandb_log['eigf_l2_error_1'] = l2_err

                    if compute_L2_error:
                        training_info["norm_sqd_diff"].append(norm_sqd_diff.detach())
                        wandb_log['l2_error'] = norm_sqd_diff.detach()

                        if norm_sqd_diff_single is not None:
                            training_info["norm_sqd_diff_single"].append(norm_sqd_diff_single.detach())
                            wandb_log['l2_error_single'] = norm_sqd_diff_single.detach()


                    if algorithm == "moment" and itr == 5000:
                        current_lr = optimizer.param_groups[-1]["lr"]
                        optimizer.param_groups[-1]["lr"] = 1e-4
                        new_lr = optimizer.param_groups[-1]["lr"]
                        print(f"current_lr: {current_lr}, new_lr: {new_lr}")
                    
                    wandb.log(wandb_log)

                    if (
                        itr % cfg.method.compute_control_objective_every
                        == cfg.method.compute_control_objective_every - 1
                        or itr == cfg.method.num_iterations - 1
                    ):
                        torch.save(solver.state_dict(), experiment_path + f'/solver_weights_{itr+1:_}.pth')
                        
                        with open(experiment_path + '/training_info.pkl', 'wb') as f:
                            pickle.dump(training_info,f)


    cleanup()  # Cleanup process group after training
    wandb.finish()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    algorithms = ['SOCM','SOCM_adjoint','rel_entropy']
    #algorithms = ["EFC"]
    world_size = min(len(algorithms),torch.cuda.device_count())
    mp.spawn(main, args=(world_size,algorithms), nprocs=world_size, join=True)
