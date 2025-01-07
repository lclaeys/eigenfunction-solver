import numpy as np

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

    def inner_prod(self, f, k=1):
        """
        Returns the inner product between the function f and the kth eigenfunction (starting from m=1)
        Args:
            f (Function): function to compute inner products with. Should allow vectorized inputs
            k (int)
        Returns
            inner_prod: inner prod between f and k-th eigenfunction
        
        """
        if self.m_computed < k:
            self.exact_eval = k
            self.eigvals = self.energy.exact_eigvals(k)
            self.eigfuncs = self.energy.exact_eigfunctions(self.samples, k)
        
        inner_prod = np.sum(f(self.samples)*self.eigfuncs[:,k-1],axis=0)

        return inner_prod

    def solve(self, f, t, x, k):
        """
        Solves the PDE with initial condition f (function) at times t and positions x
        Args:
            f (Function): initial condition
            t (array)[T]: times at which to evaluate
            x (array)[n,d]: points where to evaluate
            k (int): number of eigenfunctions to use in the approximation
        Returns:
            sol (array)[n,T]: approximate solution of PDE at points x and times t
        """

        inner_prods = []
        i = 1
        sol = np.zeros([x.shape[0],len(t)])
        while i <= k:
            inner_prod = self.inner_prod(f,i)
            inner_prods.append(inner_prod)
            sol += np.exp(self.eigvals[i-1]*t)[None,:]
        
        return sol