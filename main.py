"""

Main script for running experiments. Once again, structure is adapted from main.py in https://github.com/facebookresearch/SOC-matching.

"""
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import math
from omegaconf import OmegaConf
import time
import pickle
from copy import deepcopy

from SOC_eigf.method import SOC_Solver
from SOC_eigf.utils import compute_EMA, stochastic_trajectories, stochastic_trajectories_final, log_normalization_constant, control_objective
from SOC_eigf.experiment_settings import settings

torch.autograd.set_detect_anomaly(True)

def setup(rank, world_size):
    """Initialize process group for distributed training."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Destroy process group to free resources."""
    dist.destroy_process_group()

def main(rank, world_size, args_cfgs):
    setup(rank, world_size)
    # Set device for the current process
    print(OmegaConf.to_yaml(args_cfgs[rank]))

    device = torch.device(f"cuda:{args_cfgs[rank].gpu}")
    torch.cuda.init()
    torch.cuda.set_device(device)

    args_cfg = args_cfgs[rank]

    experiment_cfg = OmegaConf.load('experiment_cfg.yaml')

    cfg = OmegaConf.merge(experiment_cfg,args_cfg)
    torch.manual_seed(cfg.seed)

    experiment_folder = f'experiments/{cfg.experiment_name}'
    experiment_path = experiment_folder + f'/{cfg.method}/{cfg.run_name}'

    if cfg.method == "COMBINED":
        try:
            saved_eigf_cfg = OmegaConf.load(experiment_folder + f'/EIGF/{cfg.trained_eigf_run_name}/cfg.yaml')
        except FileNotFoundError as e:
            print('Error: Please train EIGF model before attempting to train combined model!')
            raise e
        
        cfg.eigf = saved_eigf_cfg.eigf

    os.makedirs(experiment_path,exist_ok=True)
    OmegaConf.save(cfg, experiment_path + '/cfg.yaml')
    
    ts = torch.linspace(0, cfg.T, cfg.num_steps + 1).to(cfg.device)

    if cfg.method=="COMBINED":
        cutoff_idx = torch.searchsorted(ts,cfg.T - cfg.ido.T_cutoff) - 1
        first_ts = ts[:cutoff_idx]
        ido_ts = ts[cutoff_idx:]

    x0, sigma, optimal_sde, neural_sde = settings.define_variables(cfg, ts)

    if optimal_sde is not None:
        ground_truth_control = optimal_sde.u
    else:
        ground_truth_control = None

    state0 = x0.repeat(cfg.optim.batch_size, 1)

    if cfg.method == "COMBINED":
        try:
            checkpoint = torch.load(experiment_folder + f'/EIGF/{cfg.trained_eigf_run_name}/neural_sde_weights.pth')
        except FileNotFoundError as e:
            print('Error: Please train EIGF model before attempting to train combined model!')
            raise e
        
        neural_sde.load_state_dict(checkpoint, strict=False)

        neural_sde.eigf_model = None

        for param in neural_sde.eigf_gs_model.parameters():
            param.requires_grad = False

        print(f'Succesfully loaded eigenfunction model {cfg.trained_eigf_run_name}.')

    #### Normalization constant ####

    if cfg.method in ['IDO','COMBINED']:
        norm_state0 = x0.repeat(65536, 1)
        log_normalization_const = log_normalization_constant(
            neural_sde,
            norm_state0,
            ts if cfg.method == "IDO" else ido_ts,
            cfg,
            n_batches_normalization=1,
            ground_truth_control=ground_truth_control,
        )

    solver = SOC_Solver(
        neural_sde,
        x0,
        ground_truth_control,
        T=cfg.T,
        num_steps=cfg.num_steps,
        lmbd=cfg.lmbd,
        d=cfg.d,
        sigma=sigma,
        solver_cfg=cfg.solver
    )

    algorithm = cfg.solver.ido_algorithm
    if algorithm == "SOCM_exp":
        solver.gamma = torch.nn.Parameter(
            torch.tensor([cfg.ido.gamma]).to(cfg.device)
        )
    else:
        solver.gamma = cfg.ido.gamma

    def initialize_optimizers():
        gs_optimizer = None
        eigf_optimizer = None
        ido_optimizer = None

        if cfg.method=="EIGF":
            gs_optimizer = optim.Adam(params=solver.neural_sde.eigf_gs_model.parameters(),
                                    lr=cfg.optim.adam_lr,
                                    eps=cfg.optim.adam_eps,
                                    weight_decay=cfg.optim.adam_wd)
            gs_optimizer.zero_grad()

            if cfg.eigf.k > 1:
                eigf_optimizer = optim.Adam(params=solver.neural_sde.eigf_model.parameters(),
                                    lr=cfg.optim.adam_lr,
                                    eps=cfg.optim.adam_eps,
                                    weight_decay=cfg.optim.adam_wd)
                eigf_optimizer.zero_grad()

        elif cfg.method in ['IDO', 'COMBINED']:
            if algorithm == "moment":
                ido_optimizer = torch.optim.Adam(
                    [{"params": solver.neural_sde.ido_model.parameters()}]
                    + [{"params": solver.y0, "lr": cfg.optim.y0_lr}],
                    lr=cfg.optim.ido_lr,
                    eps=cfg.optim.adam_eps,
                )
            elif algorithm == "SOCM_exp":
                ido_optimizer = torch.optim.Adam(
                    [{"params": solver.neural_sde.ido_model.parameters()}]
                    + [{"params": solver.gamma, "lr": cfg.optim.M_lr,"params": solver.neural_sde.gamma, "lr": cfg.optim.M_lr}],
                    lr=cfg.optim.ido_lr,
                    eps=cfg.optim.adam_eps,
                )
            elif algorithm == "SOCM":
                ido_optimizer = torch.optim.Adam(
                    [{"params": solver.neural_sde.ido_model.parameters()}]
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
                    lr=cfg.optim.ido_lr,
                    eps=cfg.optim.adam_eps,
                )
            else:
                ido_optimizer = torch.optim.Adam(
                    solver.neural_sde.ido_model.parameters(), lr=cfg.optim.ido_lr, eps=cfg.optim.adam_eps
                )

            ido_optimizer.zero_grad()

        return gs_optimizer, eigf_optimizer, ido_optimizer
        
    gs_optimizer, eigf_optimizer, ido_optimizer = initialize_optimizers()

    logged_variables=[
        "itr",
        "iteration_time",
        "loss"
    ]

    if cfg.method == "EIGF":
        logged_variables += ['main_loss', 'orth_loss']
        
        if cfg.eigf.k > 1:
            logged_variables += ['es_loss', 'es_main_loss', 'es_orth_loss']

        if cfg.setting[:2] == "OU":
            logged_variables += ["eigf_error", "grad_log_eigf_error"]
            exact_eigvals = solver.neural_sde.exact_eigvals(cfg.eigf.k)
            print(f"EXACT EIGENVALUE: {exact_eigvals}")
            solver.neural_sde.eigvals = exact_eigvals
    
    if cfg.compute_objective_every is not None:
        logged_variables += ['control_objective_mean','control_objective_std']
    
    if cfg.setting[:2] == "OU":
        logged_variables += ['control_l2_error']

    logs = {var: np.full(cfg.num_iterations,np.nan) for var in logged_variables}

    EMA_weight_mean_coeff = 0.002

    # diagnostic to assess convergence and smoothen eigenvalue computation
    if cfg.eigf.k > 1 or cfg.solver.finetune:
        eigval_returns = 0.0
        eigval_vol = 0.0
        eigval_EMA_coeff = 0.5
        compute_eigval_every = 100
        prev_stored_eigval = torch.tensor([1.0], device=x0.device,requires_grad=False)
        stored_eigval = torch.tensor([1.0], device=x0.device,requires_grad=False)
        prev_stored_eigval_1 = torch.tensor([1.0], device=x0.device,requires_grad=False)
        stored_eigval_1 = torch.tensor([1.0], device=x0.device,requires_grad=False)
        eigval_converged = False
        if cfg.solver.finetune:
            solver.eigf_loss = 'ritz'

    with torch.enable_grad():
        torch.save(solver.state_dict(), experiment_path + f'/solver_weights_0.pth')
        for itr in range(cfg.num_iterations):
            solver.train()
            start = time.time()

            verbose = itr == 0
            logs["itr"][itr] = itr

            if cfg.method == "EIGF":
                (
                    loss, 
                    main_loss, 
                    orth_loss,
                    gs_fx,
                    gs_Dfx
                ) = solver.gs_loss(
                    verbose=verbose
                )

                logs['loss'][itr] = loss.detach()
                logs['main_loss'][itr] = main_loss.detach()
                logs['orth_loss'][itr] = orth_loss.detach()

                loss.backward()

                # gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(solver.neural_sde.eigf_gs_model.parameters(), 1.0)

                gs_optimizer.step()
                gs_optimizer.zero_grad()

                es_loss = None

                if cfg.eigf.k > 1 and eigval_converged:
                    (
                    es_loss, 
                    es_main_loss, 
                    es_orth_loss,
                    fx,
                    Dfx
                    ) = solver.es_loss(
                        gs_fx,
                        gs_Dfx,
                        verbose=verbose
                    )

                    logs['es_loss'][itr] = es_loss.detach()
                    logs['es_main_loss'][itr] = es_main_loss.detach()
                    logs['es_orth_loss'][itr] = es_orth_loss.detach()

                    es_loss.backward()

                    eigf_optimizer.step()
                    eigf_optimizer.zero_grad()

            elif cfg.method == "IDO":
                (
                    loss,
                    log_weight_mean,
                    weight_std
                ) = solver.ido_loss(
                    log_normalization_const,
                    state0,
                    verbose
                )                    
                
                logs['loss'][itr] = loss.detach()

                loss.backward()

                ido_optimizer.step()
                ido_optimizer.zero_grad()

                log_normalization_const = compute_EMA(
                        log_weight_mean.detach(),
                        log_normalization_const,
                        EMA_coeff=EMA_weight_mean_coeff,
                        itr=itr,
                    )
                
            elif cfg.method == "COMBINED":
                if itr % cfg.solver.new_trajectory_every == 0:
                    intermediate_state0, _,_,_ = stochastic_trajectories_final(
                        neural_sde,
                        state0,
                        first_ts,
                        cfg.lmbd,
                        verbose=False
                    )

                solver.ts = ido_ts
                solver.num_steps = len(ido_ts)-1

                (
                    loss,
                    log_weight_mean,
                    weight_std
                ) = solver.ido_loss(
                    log_normalization_const,
                    intermediate_state0,
                    verbose
                )                    
                
                logs['loss'][itr] = loss.detach()

                loss.backward()

                ido_optimizer.step()
                ido_optimizer.zero_grad()

                log_normalization_const = compute_EMA(
                        log_weight_mean.detach(),
                        log_normalization_const,
                        EMA_coeff=EMA_weight_mean_coeff,
                        itr=itr,
                    )
                
                solver.ts = ts
                solver.num_steps = len(ts)-1

            if (cfg.solver.finetune or cfg.eigf.k > 1) and cfg.method == "EIGF" and not eigval_converged:
                if itr % compute_eigval_every == 0:
                    new_returns = stored_eigval / (prev_stored_eigval.abs() + 1e-2) * prev_stored_eigval.sign() - 1
                    eigval_returns = compute_EMA(new_returns, eigval_returns, eigval_EMA_coeff, itr//compute_eigval_every)
                    eigval_vol = compute_EMA(new_returns**2, eigval_vol, eigval_EMA_coeff, itr // compute_eigval_every) 

                    prev_stored_eigval = stored_eigval.clone()
                    stored_eigval = solver.neural_sde.eigvals[0]

                    print(f'Iteration {itr} {cfg.run_name}: eigval {prev_stored_eigval.squeeze().detach():.3E} | returns {eigval_returns.detach().squeeze():.3E} | vol {eigval_vol.detach().squeeze():.3E}')
                else:
                    i = itr % compute_eigval_every
                    stored_eigval = stored_eigval * i / (i+1) + solver.neural_sde.eigvals[0] / (i+1)
                
                if eigval_returns.abs() < 1e-2 and eigval_vol < 1e-2 and itr >= 500 and not eigval_converged:
                    eigval_converged = True

                    if cfg.solver.finetune:
                        solver.neural_sde.eigvals[0] = prev_stored_eigval.detach()
                        solver.eigf_loss = cfg.solver.eigf_loss

                        print(f'First eigval converged, starting fine-tuning with lambda = {solver.neural_sde.eigvals[0].detach():.3E}.')
                        solver.beta = 1 / solver.neural_sde.eigvals[0].detach().abs()
                        gs_optimizer, eigf_optimizer, ido_optimizer = initialize_optimizers()

                    else:
                        solver.beta = 1 / solver.neural_sde.eigvals[0].detach().abs()
                        print('First eigval converged, starting training of excited states.')

                    # copy weights of gs model into es model
                    # if cfg.eigf.k > 1:
                    #     with torch.no_grad():
                    #         for i in range(len(solver.neural_sde.eigf_gs_model.net) - 1):  # skip final layer
                    #             for target_param, source_param in zip(solver.neural_sde.eigf_model.net[i].parameters(), solver.neural_sde.eigf_gs_model.net[i].parameters()):
                    #                 target_param.copy_(source_param)
                            
                    #         source_last = solver.neural_sde.eigf_gs_model.net[-1][0]  # get nn.Linear
                    #         target_last = solver.neural_sde.eigf_model.net[-1][0]  # get nn.Linear

                    #         target_last.weight[:cfg.eigf.k-1].copy_(source_last.weight)
                    #         target_last.bias[:cfg.eigf.k-1].copy_(source_last.bias)
            
            if cfg.eigf.k > 1 and cfg.method == "EIGF":
                if itr % compute_eigval_every == 0:
                    prev_stored_eigval_1 = stored_eigval_1.clone()
                    stored_eigval_1 = solver.neural_sde.eigvals[1]
                else:
                    i = itr % compute_eigval_every
                    stored_eigval_1 = stored_eigval_1 * i / (i+1) + solver.neural_sde.eigvals[1] / (i+1)

            with torch.no_grad():
                end = time.time()
                logs['iteration_time'][itr] = end - start

                solver.eval()

                if itr % cfg.compute_objective_every == 0:
                    objective_mean, objective_std = control_objective(
                        neural_sde,
                        state0,
                        ts,
                        cfg,
                        65536,
                        65536
                    )
                    logs['control_objective_mean'][itr] = objective_mean
                    logs['control_objective_std'][itr] = objective_std
                    print(f'Iteration {itr} {cfg.run_name}: objective {objective_mean:5.6E} pm {objective_std:5.6E}')

                if cfg.setting[:2] == "OU" and itr % cfg.compute_control_error_every == 0:
                    # use trajectories from ground truth control
                    neural_sde.use_learned_control = False
                    neural_sde.u = ground_truth_control
                    (
                    states,
                    _,
                    _,
                    _,
                    _,
                    target_control
                    ) = stochastic_trajectories(
                        neural_sde,
                        state0,
                        ts.to(state0),
                        cfg.lmbd,
                        detach=True)
                    neural_sde.use_learned_control = True
                    
                    t_eval = int(len(ts) * cfg.eval_frac)
                    if t_eval == len(ts):
                        t_eval = len(ts)-1

                    learned_control = neural_sde.control(ts[:t_eval],states[:t_eval]).detach()
                    target_control = target_control[:t_eval]

                    norm_sqd_diff = torch.sum(
                            (target_control - learned_control) ** 2
                            / (target_control.shape[0] * target_control.shape[1])
                    )
                    logs['control_l2_error'][itr] = norm_sqd_diff
                    print(f'Iteration {itr} {cfg.run_name}: control l2 error {norm_sqd_diff:5.6E}')

                if cfg.method == "EIGF" and cfg.setting[:2] == "OU" and itr % cfg.log_every == 0:
                    x = solver.samples.detach()

                    exact_fx = solver.neural_sde.exact_eigfunctions(x, 1).cpu()
                    exact_grad_log_x = solver.neural_sde.exact_grad_log_gs(x).cpu()

                    predicted_fx = gs_fx.exp().detach().cpu()
                    predicted_grad_log_fx = gs_Dfx.detach().cpu()

                    #predicted_fx_norm = torch.mean(predicted_fx**2).sqrt()

                    l2_err = torch.mean((exact_fx - predicted_fx)**2)
                    
                    logs['eigf_error'][itr] = l2_err

                    l2_err_grad_log = torch.mean((exact_grad_log_x - predicted_grad_log_fx.squeeze(1))**2).sum(dim=-1).mean()

                    logs['grad_log_eigf_error'][itr] = l2_err_grad_log

                    eigval_error = (exact_eigvals[0] - solver.neural_sde.eigvals[0])**2
                    print(f'Iteration {itr} {cfg.run_name} [GROUND STATE]: error {l2_err:.3E} | gradlog error {l2_err_grad_log.detach().squeeze():.3E} | loss {loss.detach().squeeze():.3E} | orth loss {orth_loss.detach().squeeze():.3E}')
                    
                    if cfg.eigf.k > 1 and es_loss is not None:
                        print(f'Iteration {itr} {cfg.run_name} [EXCITED STATE]: loss {es_loss.detach().squeeze():.3E} | orth loss {es_orth_loss.detach().squeeze():.3E} | stored eigval 1 {prev_stored_eigval_1}')
                    

                if itr % cfg.log_every == 0:
                    df = pd.DataFrame(logs)
                    df.to_csv(experiment_path + f'/logs.csv',index=False)
            
                if (itr % cfg.save_model_every == 0):
                    if cfg.eigf.k > 1 and cfg.method == "EIGF":
                        solver.neural_sde.eigvals[1] = prev_stored_eigval_1
                    torch.save(solver.state_dict(), experiment_path + f'/solver_weights_{itr:_}.pth')
                    torch.save(solver.neural_sde.state_dict(), experiment_path + f'/neural_sde_weights.pth')
    
    cleanup()

# logic for expanding passed cfg parameters to allow multiple runs in parallel
def flatten_cfg(cfg, prefix=""):
    """
    Recursively flatten a plain dictionary.
    """
    flat = {}
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_cfg(v, full_key))
        else:
            flat[full_key] = v
    return flat

def set_nested(cfg, key_path, value):
    """
    Set a value in a nested dictionary given a dot-separated key path.
    """
    keys = key_path.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value

def expand_cfg(cfg):
    """
    Expand a config by detecting list-valued keys, verifying that all such
    lists have the same length, and creating a separate OmegaConf config for
    each index.
    """
    # Convert cfg to a plain dict with all interpolations resolved.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Flatten the plain dictionary.
    flat = flatten_cfg(cfg_dict)
    
    # Identify keys with list values.
    list_keys = [k for k, v in flat.items() if isinstance(v, list)]
    
    # If no list-valued keys, return the original cfg.
    if not list_keys:
        return [cfg]
    
    # Ensure that all list-valued keys have the same length.
    list_lengths = [len(flat[k]) for k in list_keys]
    if len(set(list_lengths)) != 1:
        raise ValueError("All list-valued arguments must have the same length.")
    
    num_configs = list_lengths[0]
    expanded_cfgs = []
    
    # For each index, create a new config where list values are replaced by their corresponding element.
    for i in range(num_configs):
        new_cfg = {}
        for k, v in flat.items():
            if k in list_keys:
                set_nested(new_cfg, k, v[i])
            else:
                set_nested(new_cfg, k, v)
        expanded_cfgs.append(OmegaConf.create(new_cfg))
        
    return expanded_cfgs

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli_cfg = OmegaConf.from_cli()

    args_cfgs = expand_cfg(cli_cfg)
    world_size = min(len(args_cfgs),torch.cuda.device_count())

    mp.spawn(main, args=(world_size,args_cfgs), nprocs=world_size, join=True)