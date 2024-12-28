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


ps = np.arange(5,20)*10
kernel_scales = np.logspace(-0.5,1.5,20)

dim = 1
k = 6

hyperparam_names = ['dim', 'k', 'p', 'scale', 'L_reg']
hyperparams = [
    [dim],
    [k],
    np.arange(1,9)*100,
    np.logspace(-1,1,10),
    [1e-6],
]

hyperparam_array = np.array(list(product(*hyperparams)))

metrics = ['eigen_error','orth_error','eigen_cost', 'eigenvalue_mse','eigenfunc_mse']

results = np.zeros((len(hyperparam_array),len(metrics)))

energy = QuadraticEnergy(np.eye(dim))
evaluator = EigenEvaluator(energy)
x_eval = energy.exact_sample((50000,))

def kernel_experiment(hyperparams_array, kernel_class = GaussianKernel):
    experiment_params = {
        hyperparam_names[i]: hyperparams_array[i] for i in range(len(hyperparam_names))
    }

    experiment_params['p'] = int(experiment_params['p'])
    experiment_params['k'] = int(experiment_params['k'])
    experiment_params['dim'] = int(experiment_params['dim'])

    np.random.seed(42)
    dim = experiment_params['dim']
    energy = QuadraticEnergy(np.eye(dim))
    x = energy.exact_sample((50000,))

    kernel = kernel_class(experiment_params)

    p = experiment_params['p']
    L_reg = experiment_params['L_reg']
    k = experiment_params['k']
    
    basis_points = energy.exact_sample((p,))
    basis = KernelBasis(kernel, basis_points)
    basis = ConstantBasis(basis)

    solver = GalerkinSolver(energy, 1.0, experiment_params)

    solver = solver.fit(x,basis,k=min(p-1,k),L_reg=L_reg)

    return x, solver

for i in tqdm(range(len(results))):
    x, solver = kernel_experiment(hyperparam_array[i], GaussianKernel)

    if solver is None:
        results[i,:] = np.nan
    else:
        out = evaluator.evaluate_metrics(solver, x_eval, metrics, k = k)
        for j, metric in enumerate(metrics):
            results[i,j] = np.sum(out[metric])

df = pd.DataFrame(np.concatenate([hyperparam_array, results],axis=1), columns = hyperparam_names + metrics)


df.to_csv('gaussian1d_1.csv',index=False)