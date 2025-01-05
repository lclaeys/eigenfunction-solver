import sys
import os
sys.path.insert(0, '/home/lclaeys/eigenfunction-solver')

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from itertools import product

from src.eigensolver.galerkin.galerkin_eigensolver import GalerkinSolver
from src.energy.quadratic import QuadraticEnergy
from src.eigensolver.galerkin.kernel_basis import KernelBasis
from src.eigensolver.galerkin.constant_basis import ConstantBasis
from src.eigensolver.galerkin.orthogonal_basis import OrthogonalBasis
from src.eigensolver.galerkin.kernels.gaussian_kernel import GaussianKernel
from src.eigensolver.galerkin.kernels.polynomial_kernel import PolynomialKernel
from src.metrics.eigen_evaluator import EigenEvaluator


ps = [400]
kernel_scales = np.logspace(0,4,20)

dim = 20
k = 300

hyperparam_names = ['dim', 'k', 'p', 'scale', 'L_reg','num_samples','order']
int_hyperparams = ['dim', 'k', 'p', 'num_samples','order']
hyperparams = [
    [dim],
    [k],
    ps,
    kernel_scales,
    [1e-6],
    [50000],
    [3,4,5]
]

hyperparam_array = np.array(list(product(*hyperparams)))

metrics = ['eigen_error','orth_error','eigen_cost', 'fitted_eigen_error','eigenvalue_mse', 'fitted_eigenvalue_mse','eigenfunc_mse', 'linear_reconstruction', 'L_linear_reconstruction', 'quadratic_reconstruction', 'L_quadratic_reconstruction']

def kernel_experiment(hyperparams_array, energy, kernel_class = PolynomialKernel):
    experiment_params = {
        hyperparam_names[i]: hyperparams_array[i] for i in range(len(hyperparam_names))
    }

    for param in int_hyperparams:
        experiment_params[param] = int(experiment_params[param])

    np.random.seed(42)
    dim = experiment_params['dim']
    x = energy.exact_sample((1000000,))

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

for n in range(10):
    
    experiment_name = f'randomdiag/20d_polynomial_{n}'
    print(f'Starting experiment {experiment_name}')
    np.random.seed(n)

    A = np.diag(np.random.random(dim)*1.8+0.2)
    params = {'A': A}
    
    np.savez(f'{experiment_name}.npz', **params)

    energy = QuadraticEnergy(A)
    evaluator = EigenEvaluator(energy)
    x_eval = energy.exact_sample((50000,))

    results = np.zeros((len(hyperparam_array),len(metrics),k))

    for i in tqdm(range(len(results))):
        x, solver = kernel_experiment(hyperparam_array[i], energy, GaussianKernel)

        if solver is None:
            results[i,:,:] = np.nan
        else:
            out = evaluator.evaluate_metrics(solver, x_eval, metrics, k = k)
            for j, metric in enumerate(metrics):
                results[i,j,:] = out[metric]

    dfs = []
    for i in range(k):
        df = pd.DataFrame(np.concatenate([hyperparam_array, results[:,:,i]],axis=1), columns = hyperparam_names + metrics)
        df['k'] = i+1
        dfs.append(df)

    df = pd.concat(dfs)

    df.to_csv(f'{experiment_name}.csv',index=False)