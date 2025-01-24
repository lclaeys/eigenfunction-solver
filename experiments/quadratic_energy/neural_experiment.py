import sys
import os
sys.path.insert(0, '/home/lclaeys/eigenfunction-solver')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from src.eigensolver.neural.neural_eigensolver import NeuralSolver
from src.energy.quadratic import QuadraticEnergy
from src.energy.gaussian_mixture import GaussianMixture
from src.eigensolver.galerkin.kernel_basis import KernelBasis
from src.eigensolver.neural.network.feedforward import FeedForwardNetwork, ConstantFFN
from src.eigensolver.neural.loss.orth_loss import BasicOrthogonalityLoss, CovOrthogonalityLoss
from src.eigensolver.neural.loss.variational_loss import VariationalLoss
from torch.utils.data import Dataset, DataLoader

from src.pdesolver.fitted_solver import FittedEigenSolver

from src.metrics.eigen_evaluator import EigenEvaluator
from src.metrics.pde_evaluator import ExactPDEEvaluator
from src.metrics.reconstruction_evaluator import ReconstructionEvaluator
from torch.optim.lr_scheduler import LambdaLR

dim = 20

metrics = ['eigenvalue_mse','eigenfunc_mse','orth_error']


# standard metrics
metrics = ['orth_error',
           'eigenvalue_mse', 
           'eigenfunc_mse',
           "fitted_eigenvalue_mse",
           'eigen_error',
           'fitted_eigen_error']

# functions for reconstruction error
funcs = {
    'linear': lambda x: x.sum(),
    'quadratic': lambda x: (x**2).sum(),
    'cubic': lambda x: (x**3).sum()
}

# inner products for PDE error
linear_mult = dim
quadratic_mult = ((dim+1)*dim)//2
pdes = {
    'linear': np.array([0] + [1]),
    'quadratic': np.array([0] * (linear_mult + 1) + [1]),
    "cubic": np.array([0]*(quadratic_mult + linear_mult + 1)+[1])
}

print(pdes)

energy = QuadraticEnergy(np.eye(dim))

eigen_evaluator = EigenEvaluator(energy)

reconstruction_evaluators = {func_name: ReconstructionEvaluator(energy, funcs[func_name])
                             for func_name in funcs}

x_eval = energy.exact_sample((50000,))

def combined_scheduler(step, warmup_steps, total_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps  # Linear warmup
    # Cosine decay after warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return max_lr / 2 * (1 + np.cos(np.pi * progress))

def train():
    run = wandb.init()
    config = wandb.config

    k = config.k
    dim = config.dim

    layer_size = config.layer_size
    num_samples = config.num_samples
    num_layers = config.num_layers

    model = ConstantFFN([dim] + [layer_size]*num_layers +[k])

    np.random.seed(42)
    x = energy.exact_sample((num_samples,))

    num_epochs = config.epochs
    max_lr = config.max_lr
    warmup_steps = config.warmup_steps
    #momentum = config.momentum

    optimizer = optim.Adam(model.parameters(), lr = 1.0) # ADAM
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: combined_scheduler(step, warmup_steps, num_epochs, max_lr))

    solver = NeuralSolver(energy, x, model, optimizer, config, scheduler=scheduler)

    eigen_evaluator = EigenEvaluator(energy)
    reconstruction_evaluators = {func_name: ReconstructionEvaluator(energy, funcs[func_name])
                                for func_name in funcs}
    
    x_eval = energy.exact_sample((50000,))

    for epoch in range(1,num_epochs+1):
        loss = solver.train_epoch()
        solver.compute_eigfuncs()
        fitted_solver = FittedEigenSolver(energy, x_eval, solver)

        if epoch % 5 == 0 or epoch == num_epochs:
            pde_evaluator = ExactPDEEvaluator(energy, fitted_solver)

            out = eigen_evaluator.evaluate_metrics(solver, x_eval, metrics, k = k)
            for metric in metrics:
                out[metric] = out[metric][-1]
            
            fx_eval = solver.predict(x_eval)
            for j, func_name in enumerate(funcs):
                rec_error, L_rec_error = reconstruction_evaluators[func_name].compute_reconstruction_error(x_eval,fx_eval, x_eval, fx_eval, solver.fitted_eigvals)
                out[func_name + '_reconstruction'], out[func_name + '_L_reconstruction'] = rec_error[-1], L_rec_error[-1]

            for j, pde in enumerate(pdes):
                out[pde + '_pde_error'] = pde_evaluator.compute_pde_error(pdes[pde], x_eval, np.array([1]))[-1,0]

            out['loss'] = loss
            wandb.log(out)

    run.finish()

k = 250

sweep_config = {
    "method": "bayes",
    "name": "overnight_sweep_tanh_20d",
    "metric": {"name": "fitted_eigenvalue_mse", "goal": "minimize"},
    "parameters": {
        "dim": {"value": dim},
        "k": {"value": k},
        "max_lr": {"min": 1e-5, "max": 5e-2},
        "warmup_steps": {"min": 10, "max": 50},
        "momentum": {"min": 0.2, "max": 0.9},
        "epochs": {"min": 50, "max": 300},
        "beta": {"min": 0.2, "max": 0.6},
        "batch_size": {"values": [2**12,2**13]},
        "layer_size": {"min": 16, "max": 1000},
        "num_layers": {"min": 2, "max": 10},
        "num_samples": {"values": [2**13,2**14,2**15,2**16,2**17,2**18,2**19,2**20]},
        "device": {"value": "cuda"}
    },
    "total_runs": 100
}

sweep_id = wandb.sweep(sweep_config, project="neural-eigenfunction-learner")

wandb.agent(sweep_id, function=train)
