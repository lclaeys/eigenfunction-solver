import sys
import os
import argparse

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
import argparse

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
from src.soc.utils import exact_eigfunctions, log_normalization_constant, compute_EMA_log

from src.soc.method import EigenSolver, CombinedSDE, CombinedSolver

from src.experiment_settings.socm_settings.settings import define_variables as define_variables_socm
from src.experiment_settings.efc_settings.settings import define_variables as define_variables_efc

# Code essentially copied from SOC_matching main.py

def setup(rank, world_size):
    """Initialize process group for distributed training."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Destroy process group to free resources."""
    dist.destroy_process_group()

def main(rank, world_size, algorithms, gpus, args_cfg):
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
    device = torch.device(f"cuda:{gpus[rank]}")
    torch.cuda.init()
    torch.cuda.set_device(device)
    algorithm = algorithms[rank]

    experiment_cfg = OmegaConf.load('experiment_config.yaml')

    efc_cfg = OmegaConf.load('efc_config.yaml')
    socm_cfg = OmegaConf.load('socm_config.yaml')

    efc_cfg = OmegaConf.merge(experiment_cfg, efc_cfg)
    socm_cfg = OmegaConf.merge(experiment_cfg, socm_cfg)
    efc_cfg = OmegaConf.merge(efc_cfg, args_cfg)
    socm_cfg = OmegaConf.merge(socm_cfg, args_cfg)
    if experiment_cfg.method.combine:
        cfg = OmegaConf.merge(efc_cfg, socm_cfg)
    elif algorithm == "EFC":
        cfg = efc_cfg
    else:
        cfg = socm_cfg
    cfg.method.algorithm = algorithm
    name = algorithm + "_COMBINED" if cfg.method.combine else algorithm
    print(cfg.experiment.name)
    wandb.init(
        project="neural-eigenfunction-learner",
        group = cfg.experiment.name,
        name = name,
        config=dict(cfg),
        job_type='eval'
    )

    torch.manual_seed(cfg.method.seed)
    cfg.method.device = "cuda"

    appendix = ""
    if algorithm == "EFC" and not cfg.method.combine:
        appendix = f'_{cfg.method.k}'
        if cfg.arch.prior is not None:
            appendix += f'_{cfg.arch.prior}'
    
    if cfg.experiment.appendix is not None:
        appendix += cfg.experiment.appendix

    experiment_folder = f'experiments/{cfg.experiment.name}'
    experiment_path = experiment_folder + f'/{name}' + appendix

    os.makedirs(experiment_path,exist_ok=True)
    OmegaConf.save(cfg, experiment_path + '/cfg.yaml')
    OmegaConf.save(efc_cfg, experiment_path + '/efc_cfg.yaml')
    OmegaConf.save(socm_cfg, experiment_path + '/socm_cfg.yaml')

    # if algorithm != "EFC" and not cfg.method.combine and cfg.experiment.appendix == "":
    #     cfg.method.T = cfg.method.train_T
    #     cfg.method.num_steps = cfg.method.train_steps

    ts = torch.linspace(0, cfg.method.T, cfg.method.num_steps + 1).to(cfg.method.device)
    cutoff_idx = torch.searchsorted(ts,cfg.method.T - cfg.method.cutoff_T) - 1
    first_ts = ts[:cutoff_idx]
    last_ts = ts[cutoff_idx:]
    
    compute_optimal = not os.path.exists(experiment_folder + '/optimal_sde.pkl')

    if cfg.method.setting == "double_well":
        compute_optimal = False

    if algorithm == "EFC":
        x0, sigma, optimal_sde, neural_sde = define_variables_efc(cfg, ts, compute_optimal)
        log_normalization_const = 0
    elif not cfg.method.combine:
        x0, sigma, optimal_sde, neural_sde, u_warm_start = define_variables_socm(cfg, ts, compute_optimal)
    else:
        x0, sigma, optimal_sde, eigen_sde = define_variables_efc(efc_cfg, ts, compute_optimal)
        x0, sigma, optimal_sde, socm_sde, u_warm_start = define_variables_socm(socm_cfg, ts, compute_optimal)

        try:
            checkpoint = torch.load(experiment_folder + f'/EFC_2/solver_weights_80_000.pth')
        except FileNotFoundError as e:
            print('Error: Please train EFC model before attempting to train combined model!')
            raise e

        eigen_sde_dict = {}
        for key in checkpoint.keys():
            if key[:10] == "neural_sde":
                eigen_sde_dict[key[11:]] = checkpoint[key]
        eigen_sde.load_state_dict(eigen_sde_dict)

        neural_sde = CombinedSDE(socm_sde,
                                 eigen_sde, 
                                 cutoff_T = cfg.method.cutoff_T,
                                 train_value=cfg.method.train_value,
                                 use_terminal=cfg.method.use_terminal)
        
        # Freeze parameters of eigensolver
        neural_sde.initialize_models()

        for param in neural_sde.eigen_sde.parameters():
            param.requires_grad = False

    if cfg.method.load_state:
        checkpoint = torch.load(experiment_folder + f'/{algorithm}_T1/solver_weights_80_000.pth')
        neural_sde_dict = {}
        checkpoint['neural_sde.nabla_V.down_0.0.weight'][:,0] /= cfg.method.T
        checkpoint['neural_sde.nabla_V.res_0.0.weight'][:,0] /= cfg.method.T

        for key in checkpoint.keys():
            if key[:10] == "neural_sde":
                neural_sde_dict[key[11:]] = checkpoint[key]
                
        neural_sde.load_state_dict(neural_sde_dict)
        print('Loaded initialization from T=1 solution.')

    if compute_optimal or cfg.method.setting == "double_well":
        with open(experiment_folder + '/optimal_sde.pkl','wb') as file:
            pickle.dump(optimal_sde, file)
        print('Computed and saved ground truth solution.')
    else:
        with open(experiment_folder + '/optimal_sde.pkl','rb') as file:
            optimal_sde = pickle.load(file)
            if optimal_sde is not None:
                 optimal_sde = optimal_sde.to(device)

        if cfg.method.setting == "double_well":
            optimal_sde.u.ut = optimal_sde.u.ut.to(device)
        elif cfg.method.setting[:2] == "OU":
            optimal_sde.u.u = optimal_sde.u.u.to(device)

        print('Loaded ground truth solution.')
    
    if optimal_sde is not None:
        ground_truth_control = optimal_sde.u
    else:
        ground_truth_control = None

    state0 = x0.repeat(cfg.optim.batch_size, 1)

    if algorithm != "EFC" and rank == 0:
        ########### Compute normalization constant and control L2 error for initial control ############
        print(
            f"Estimating normalization constant for initial control..."
        )
        norm_ts = last_ts if cfg.method.combine else ts

        (
            log_normalization_const
        ) = log_normalization_constant(
            neural_sde,
            state0,
            norm_ts,
            cfg,
            n_batches_normalization=512,
            ground_truth_control=ground_truth_control,
        )
        print(
            f"log_normalization_constant: {log_normalization_const}"
        )

    log_normalization_const = torch.tensor(log_normalization_const if rank == 0 else 0.0, device=device)
    dist.broadcast(log_normalization_const, src=0)

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
            langevin_dt=cfg.method.langevin_dt,
            beta=cfg.loss.beta
        )
    elif not cfg.method.combine:
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
    else:
        solver = CombinedSolver(
            neural_sde,
            x0,
            T=cfg.method.T,
            num_steps=cfg.method.num_steps,
            lmbd=cfg.method.lmbd,
            d=cfg.method.d,
            sigma=sigma
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

    elif not cfg.method.combine:
        if algorithm == "moment":
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
    else:
        if algorithm == "moment":
            optimizer = torch.optim.Adam(
                [{"params": solver.neural_sde.neural_sde.parameters()}]
                + [{"params": solver.y0, "lr": cfg.optim.y0_lr}]
                + [{"params": solver.neural_sde.epsilon_param}]
                ,
                lr=cfg.optim.nabla_V_lr,
                eps=cfg.optim.adam_eps,
            )
        elif algorithm == "SOCM_exp":
            optimizer = torch.optim.Adam(
                [{"params": solver.neural_sde.neural_sde.parameters()}]
                + [{"params": solver.gamma, "lr": cfg.optim.M_lr}]
                + [{"params": solver.neural_sde.epsilon_param}]
                ,
                lr=cfg.optim.nabla_V_lr,
                eps=cfg.optim.adam_eps,
            )
        elif algorithm == "SOCM":
            if cfg.method.use_stopping_time:
                optimizer = torch.optim.Adam(
                    [{"params": solver.neural_sde.neural_sde.model.parameters()}]
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
                    ]
                    + [{"params": solver.neural_sde.epsilon_param}]
                    ,
                    lr=cfg.optim.nabla_V_lr,
                    eps=cfg.optim.adam_eps,
                )
            else:
                optimizer = torch.optim.Adam(
                    [{"params": solver.neural_sde.neural_sde.model.parameters()}]
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
                    + [{"params": solver.neural_sde.epsilon_param}]
                    ,
                    lr=cfg.optim.nabla_V_lr,
                    eps=cfg.optim.adam_eps,
                )
        elif algorithm == "rel_entropy":
            optimizer = torch.optim.Adam(
                [{'params': solver.neural_sde.neural_sde.model.parameters()}]
                + [{"params": solver.neural_sde.epsilon_param}]
                ,
                lr=cfg.optim.nabla_V_lr, 
                eps=cfg.optim.adam_eps
            )
        else:
            optimizer = torch.optim.Adam(
                [{'params': solver.neural_sde.neural_sde.model.parameters()}]
                + [{"params": solver.neural_sde.epsilon_param}]
                , lr=cfg.optim.nabla_V_lr, eps=cfg.optim.adam_eps
            )
        
        # warmup_epochs = cfg.optim.warmup_epochs
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda = lambda epoch: ((epoch + 1) / warmup_epochs) if epoch < warmup_epochs else
        #                             0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (cfg.method.num_iterations - warmup_epochs)))
        #     )

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

    #compute_L2_error = ground_truth_control is not None
    compute_L2_error = False
    compute_eigen_error = False

    EMA_weight_mean_coeff = 0.002

    #energy = get_energy(cfg, neural_sde)
    wandb_log = {}

    ###### Train control ######
    with torch.inference_mode(False):
        with torch.enable_grad():
            torch.save(solver.state_dict(), experiment_path + f'/solver_weights_0.pth')
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

                compute_init = (
                    itr == 0
                    or itr % cfg.method.compute_init_every == cfg.method.compute_init_every - 1
                ) and cfg.method.combine

                if compute_init:
                    init_state,_,_,_,_,_,_,_ = stochastic_trajectories(
                        neural_sde,
                        state0,
                        first_ts.to(state0),
                        lmbd=cfg.method.lmbd,
                        detach=True
                    )
                    init_state = init_state[-1]
                    print('Computed new initial trajectory.')

                verbose = itr == 0

                if algorithm == "EFC":
                    (
                    loss,
                    var_loss,
                    orth_loss
                    ) = solver.loss(
                        sample_size=cfg.method.n_samples_loss,
                        verbose=True,
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
                elif not cfg.method.combine:
                    (
                    loss,
                    _,
                    control_objective_mean,
                    control_objective_std_err,
                    trajectory,
                    log_weight_mean,
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
                        log_normalization_const=log_normalization_const
                    )
                    
                    wandb_log['loss'] = loss.detach()
                
                else:
                    solver.ts = last_ts
                    (
                    loss,
                    log_weight_mean,
                    weight_std,
                    stop_indicators,
                    ) = solver.loss(
                        cfg.optim.batch_size,
                        algorithm=algorithm,
                        total_n_samples=cfg.method.n_samples_control,
                        verbose=verbose,
                        use_stopping_time=cfg.method.use_stopping_time,
                        log_normalization_const=log_normalization_const,
                        state0 = init_state
                    )
                    solver.ts = ts
                    
                    wandb_log['loss'] = loss.detach()
                
                # Backward pass
                if algorithm != "EFC" or cfg.arch.joint:
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

                    log_normalization_const = compute_EMA(
                        log_weight_mean.detach(),
                        log_normalization_const,
                        EMA_coeff=EMA_weight_mean_coeff,
                        itr=itr,
                    )

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

                    # if algorithm != "EFC":
                    #     normalization_const = compute_EMA(
                    #         weight_mean.detach(),
                    #         normalization_const,
                    #         EMA_coeff=EMA_weight_mean_coeff,
                    #         itr=itr,
                    #     )

                    training_info["time_per_iteration"].append(time_per_iteration)
                    training_info["loss"].append(loss.detach())
                    
                    if itr % 100 == 0 and compute_L2_error:
                        solver.eval()

                        # if algorithm == "EFC":
                        #     solver.compute_eigvals(beta=cfg.loss.beta)
                            
                        #     solver.neural_sde.compute_inner_prods(samples = solver.samples)

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
                        print(f"{itr}: l2 error {norm_sqd_diff.detach():5.6E} | log_normalization_const {log_normalization_const.detach()}")
                        norm_sqd_diff_single = None
                        if algorithm == "EFC" and cfg.method.k > 1:
                            learned_control = neural_sde.control(ts[:eval_idx],states,1)
                            norm_sqd_diff_single = torch.sum(
                                (target_control - learned_control) ** 2
                                / (target_control.shape[0] * target_control.shape[1])
                            )
                            print(f"{itr}: l2 error {norm_sqd_diff.detach():5.6E}")
                                                    

                    if compute_L2_error:
                        training_info["norm_sqd_diff"].append(norm_sqd_diff.detach())
                        wandb_log['l2_error'] = norm_sqd_diff.detach()

                        if norm_sqd_diff_single is not None:
                            training_info["norm_sqd_diff_single"].append(norm_sqd_diff_single.detach())
                            wandb_log['l2_error_single'] = norm_sqd_diff_single.detach()

                    if compute_eigen_error and itr % 100 == 0:
                        x = solver.samples

                        exact_fx = exact_eigfunctions(x, 2, neural_sde, cfg, return_grad=False).cpu()

                        fx = solver.neural_sde.eigf_models[0](x).detach().cpu().squeeze(1)
                        norm = torch.mean(fx**2).sqrt().detach().cpu()
                        
                        eigf_sq_diff = torch.mean((exact_fx[:,0]-fx/norm)**2)
                        neg_eigf_sq_diff = torch.mean((exact_fx[:,0]+fx/norm)**2)
                        exact_sq_sum = torch.mean(exact_fx[:,0]**2)

                        l2_err = min(eigf_sq_diff,neg_eigf_sq_diff) / exact_sq_sum
                        wandb_log['eigf_l2_error_0'] = l2_err
                        
                        fx = solver.neural_sde.eigf_models[1](x).detach().cpu().squeeze(1)
                        norm = torch.mean(fx**2).sqrt().detach().cpu()
                        
                        eigf_sq_diff = torch.mean((exact_fx[:,1]-fx/norm)**2)
                        neg_eigf_sq_diff = torch.mean((exact_fx[:,1]+fx/norm)**2)
                        exact_sq_sum = torch.mean(exact_fx[:,1]**2)

                        l2_err = min(eigf_sq_diff,neg_eigf_sq_diff) / exact_sq_sum
                        wandb_log['eigf_l2_error_1'] = l2_err

                        wandb_log['a'] = solver.a
                        wandb_log['b'] = solver.b
                        wandb_log['lambda_0'] = solver.neural_sde.eigvals[0]
                        wandb_log['lambda_1'] = solver.neural_sde.eigvals[1]

                        print(f'Itr {itr+1}: error {wandb_log['eigf_l2_error_0']}, eigvals {solver.neural_sde.eigvals}')

                    if algorithm == "moment" and itr == 5000:
                        current_lr = optimizer.param_groups[-1]["lr"]
                        optimizer.param_groups[-1]["lr"] = 1e-4
                        new_lr = optimizer.param_groups[-1]["lr"]
                        print(f"current_lr: {current_lr}, new_lr: {new_lr}")
                    
                    if (
                        itr % cfg.method.compute_control_objective_every
                        == cfg.method.compute_control_objective_every - 1
                        or itr == cfg.method.num_iterations - 1
                        or itr == 0
                    ):
                        torch.save(solver.state_dict(), experiment_path + f'/solver_weights_{itr+1:_}.pth')
                        
                        objective_mean, objective_std = control_objective(neural_sde, x0, ts, cfg.method.lmbd, 128, total_n_samples=cfg.method.objective_samples)

                        wandb_log['objective_mean'] = objective_mean.detach()

                        training_info["control_objective_mean"].append(
                            objective_mean.detach()
                        )
                        training_info["control_objective_std_err"].append(
                            objective_std.detach()
                        )
                        training_info["control_objective_itr"].append(itr + 1)

                        with open(experiment_path + '/training_info.pkl', 'wb') as f:
                            pickle.dump(training_info,f)
                        print(f'Iteration {itr+1} objective: {objective_mean.detach()}')
                    wandb.log(wandb_log)


    cleanup()  # Cleanup process group after training
    wandb.finish()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    args_cfg = OmegaConf.from_cli()

    algorithms = args_cfg.algorithms
    gpus = args_cfg.gpus
    world_size = min(len(algorithms),torch.cuda.device_count())
    mp.spawn(main, args=(world_size,algorithms,gpus,args_cfg), nprocs=world_size, join=True)
