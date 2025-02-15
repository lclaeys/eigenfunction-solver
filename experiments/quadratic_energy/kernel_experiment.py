import sys
import os
sys.path.insert(0, '/home/lclaeys/eigenfunction-solver')

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from itertools import product
import torch
import math

from src.energy.quadratic import QuadraticEnergy

from src.eigensolver.galerkin.galerkin_eigensolver import GalerkinSolver
from src.eigensolver.galerkin.kernel_basis import KernelBasis
from src.eigensolver.galerkin.constant_basis import ConstantBasis
from src.eigensolver.galerkin.kernels.gaussian_kernel import GaussianKernel
from src.eigensolver.galerkin.kernels.polynomial_kernel import PolynomialKernel

from src.metrics.eigen_evaluator import EigenEvaluator
from src.metrics.pde_evaluator import ExactPDEEvaluator
from src.metrics.reconstruction_evaluator import ReconstructionEvaluator

from src.pdesolver.fitted_solver import FittedEigenSolver

ps = [400]
kernel_scales = torch.logspace(0,10,20)

dim = 2
k = 10

hyperparam_names = ['dim', 'k', 'p', 'scale', 'L_reg','num_samples','batch_size']
int_hyperparams = ['dim', 'k', 'p', 'num_samples','batch_size']
hyperparams = [
    [dim],
    [k],
    ps,
    kernel_scales,
    [1e-6],
    2**np.arange(7,19),
    [2**15]
]

hyperparam_array = np.array(list(product(*hyperparams)))

experiment_name = f'galerkin_2d_corr'

A = torch.linalg.inv(torch.tensor([[1,0.95],[0.95,2.0]]))

# standard metrics
metrics = ['eigen_error',
           'orth_error',
           'fitted_eigen_error',
           'eigenvalue_mse', 
           'fitted_eigenvalue_mse',
           'eigenfunc_mse']

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
    'linear': torch.tensor([0] + [1],dtype=torch.float32),
    'quadratic': torch.tensor([0] * (linear_mult + 1) + [1],dtype=torch.float32),
    "cubic": torch.tensor([0]*(quadratic_mult + linear_mult + 1)+[1],dtype=torch.float32)
}

print(pdes)

energy = QuadraticEnergy(A)
eigen_evaluator = EigenEvaluator(energy)
reconstruction_evaluators = {func_name: ReconstructionEvaluator(energy, funcs[func_name])
                             for func_name in funcs}

def kernel_experiment(hyperparams_array, energy, kernel_class = PolynomialKernel):
    experiment_params = {
        hyperparam_names[i]: hyperparams_array[i] for i in range(len(hyperparam_names))
    }

    for param in int_hyperparams:
        experiment_params[param] = int(experiment_params[param])

    dim = experiment_params['dim']

    kernel = kernel_class(experiment_params)

    p = experiment_params['p']
    L_reg = experiment_params['L_reg']
    k = experiment_params['k']
    
    basis_points = energy.exact_sample((p,))
    basis = KernelBasis(kernel, basis_points)
    basis = ConstantBasis(basis)

    solver = GalerkinSolver(energy, x, experiment_params)

    solver = solver.fit(basis,k=min(p-1,k),L_reg=L_reg)

    return x, solver

for seed in range(100):
    print(f'Starting experiment with seed {seed:04}...')
    torch.manual_seed(seed)
    x = energy.exact_sample((2**21,))
    x_eval = energy.exact_sample((2**14,))

    metric_results = np.zeros((len(hyperparam_array),len(metrics),k))
    reconstruction_results = np.zeros((len(hyperparam_array),len(funcs),k))
    L_reconstruction_results = np.zeros((len(hyperparam_array),len(funcs),k))

    pde_results = np.zeros((len(hyperparam_array),len(pdes),k))

    for i in tqdm(range(len(hyperparam_array))):
        x, solver = kernel_experiment(hyperparam_array[i], energy, GaussianKernel)

        if solver is None:
            metric_results[i,:,:] = np.nan
            reconstruction_results[i,:,:] = np.nan
            pde_results[i,:,:] = np.nan
        else:
            out = eigen_evaluator.evaluate_metrics(solver, x_eval, metrics, k = k)
            for j, metric in enumerate(metrics):
                metric_results[i,j,:] = out[metric]
            
            fx_eval = solver.predict(x_eval)
            for j, func_name in enumerate(funcs):
                reconstruction_results[i,j,:], L_reconstruction_results[i,j,:] = reconstruction_evaluators[func_name].compute_reconstruction_error(x_eval,fx_eval, x_eval, fx_eval, solver.fitted_eigvals)
            
            fitted_solver = FittedEigenSolver(energy, x_eval, solver)
            pde_evaluator = ExactPDEEvaluator(energy, fitted_solver)

            for j, pde in enumerate(pdes):
                pde_results[i,j,:] = pde_evaluator.compute_pde_error(pdes[pde], x_eval, torch.tensor([1]))[:,0]

    dfs = []
    for i in range(k):
        hyperparam_df = pd.DataFrame(hyperparam_array,  columns = hyperparam_names)
        metric_df = pd.DataFrame(metric_results[:,:,i], columns = metrics)
        reconstruction_df = pd.DataFrame(reconstruction_results[:,:,i], columns = [func_name + '_reconstruction' for func_name in funcs])
        L_reconstruction_df = pd.DataFrame(L_reconstruction_results[:,:,i], columns = [func_name + '_L_reconstruction' for func_name in funcs])
        pde_df = pd.DataFrame(pde_results[:,:,i], columns = [pde_name + '_pde_error' for pde_name in pdes])
        
        df = pd.concat([hyperparam_df,metric_df,reconstruction_df,L_reconstruction_df,pde_df],axis=1)
        df['k'] = i+1
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(f'{experiment_name}/{seed:04}.csv',index=False)


# for n in range(10):
    
#     experiment_name = f'randomdiag/20d_polynomial_{n}'
#     print(f'Starting experiment {experiment_name}')
#     np.random.seed(n)

#     A = np.diag(np.random.random(dim)*1.8+0.2)
#     params = {'A': A}
    
#     np.savez(f'{experiment_name}.npz', **params)

#     energy = QuadraticEnergy(A)
#     evaluator = EigenEvaluator(energy)
#     x_eval = energy.exact_sample((50000,))

#     results = np.zeros((len(hyperparam_array),len(metrics),k))

#     for i in tqdm(range(len(results))):
#         x, solver = kernel_experiment(hyperparam_array[i], energy, GaussianKernel)

#         if solver is None:
#             results[i,:,:] = np.nan
#         else:
#             out = evaluator.evaluate_metrics(solver, x_eval, metrics, k = k)
#             for j, metric in enumerate(metrics):
#                 results[i,j,:] = out[metric]

#     dfs = []
#     for i in range(k):
#         df = pd.DataFrame(np.concatenate([hyperparam_array, results[:,:,i]],axis=1), columns = hyperparam_names + metrics)
#         df['k'] = i+1
#         dfs.append(df)

#     df = pd.concat(dfs)

#     df.to_csv(f'{experiment_name}.csv',index=False)