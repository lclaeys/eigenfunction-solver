import torch

class ExactEigenSolver():
    """
    Class for solving the PDE by approximating the initial condition using the exact eigenfunctions
    """
    def __init__(self, energy, samples):
        """
        Args:
            energy (BaseEnergy): energy object
            samples (ndarray): samples used to compute inner product
        """
        self.dim = energy.dim
        self.samples = samples
        self.energy = energy

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

        self.eigvals = self.energy.exact_eigvals(k)

        # (N,k)
        self.sample_fx = self.energy.exact_eigfunctions(self.samples, k)
        self.fx = self.energy.exact_eigfunctions(x, k)

        # (N,)
        self.sample_funcx = func(self.samples)

        # (k,)
        inner_prods = torch.mean(self.sample_fx*self.sample_funcx[:,None],dim=0)
        
        # sum over first index of (k,n,T)
        sol = torch.sum(inner_prods[:,None,None] * 
                     torch.exp(-self.eigvals[:,None,None]*t[None,None,:]) * 
                     (self.fx.T)[:,:,None],
                     dim = 0)
        
        return sol