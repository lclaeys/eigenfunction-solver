import numpy as np

class FittedEigenSolver():
    """
    Class for solving the PDE by approximating the initial condition using fitted eigenfunctions
    """
    def __init__(self, energy, samples, solver):
        """
        Args:
            energy (BaseEnergy): energy object
            samples (ndarray): samples used to compute inner product
            solver (BaseSolver): solver object
        """
        self.dim = energy.dim
        self.samples = samples
        self.energy = energy
        self.solver = solver

    def solve(self, func, t, x, k):
        """
        Solves the PDE with initial condition func (function) at times t and positions x
        Args:
            func (Function): initial condition
            t (array)[T]: times at which to evaluate
            x (array)[n,d]: points where to evaluate
            k (int): number of eigenfunctions to use in the approximation
        Returns:
            sol (array)[n,T]: approximate solution of PDE at points x and times t
        """

        self.eigvals = self.solver.fit_eigvals(k)

        # (N,k)
        self.sample_fx = self.solver.predict(self.samples, k)
        self.fx = self.solver.predict(x, k)

        # (N,)
        self.sample_funcx = func(self.samples)

        # (k,)
        inner_prods = np.mean(self.sample_fx*self.sample_funcx[:,None],axis=0)
        
        # sum over first index of (k,n,T)
        sol = np.sum(inner_prods[:,None,None] * 
                     np.exp(-self.eigvals[:,None,None]*t[None,None,:]) * 
                     (self.fx.T)[:,:,None],
                     axis = 0)
        
        return sol