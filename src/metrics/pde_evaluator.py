import numpy as np
import torch
import matplotlib.pyplot as plt

class ExactPDEEvaluator():
    """
    Class for evaluating the quality of the learned eigenfunction by seeing how well it solves the PDE for an initial condition
    which is a finite linear combination of eigenfunctions (and hence the exact solution is known)
    """
    def __init__(self, energy, pde_solver):
        self.energy = energy
        self.pde_solver = pde_solver
        
    def compute_pde_error(self, inner_prods, x, t):
        """
        Compute error of solving PDE with initial condition given by inner_prods@eigfuncs

        Args:
            inner_prods (array): determines initial value of PDE
            x (array): points on which to evaluate
            t (array): times at which to evaluate
        Returns:
            errs (array) [k, t]: relative MSE error of using progressively more eigenfuctions at times t
        """

        k = len(inner_prods)

        self.exact_sol = np.sum(inner_prods[:,None,None] *
                    np.exp(-self.energy.exact_eigvals(k)[:,None,None]*t[None,None,:]) * 
                    (self.energy.exact_eigfunctions(x,k).T)[:,:,None],axis=0)

        def func(x_batch):
            return inner_prods@(self.energy.exact_eigfunctions(x_batch,k)).T

        fitted_k = len(self.pde_solver.solver.fitted_eigvals)
        approx_sol = self.pde_solver.solve(func, t, x, fitted_k)

        return np.mean((approx_sol - self.exact_sol[None,:,:])**2,axis=1) / np.mean(self.exact_sol**2,axis=0)[None,:]

