import numpy as np

class ExactSolver():
    def __init__(self, energy):
        self.dim = energy.dim
        self.samples = self.energy.exact_samples(50000)
        self.m_computed = 0
        self.x = None

    def inner_prod(self, f, m=1):
        """
        Returns the inner product between the function f and the mth eigenfunction (starting from m=1)
        Args:
            f (Function): function to compute inner products with. Should allow vectorized inputs
        Returns
            inner_prod: inner prod between f and mth eigenfunction
        
        """
        if self.m_computed < m:
            self.exact_eval = m
            self.eigvals = self.energy.exact_eigvals(m)
            self.eigfuncs = self.energy.exact_eigfunctions(self.samples, m)
        
        inner_prod = np.sum(f(self.samples)*self.eigfuncs[:,m-1],axis=0)

        return inner_prod

    def solve(self, f, t, x, eps = 1e-6):
        """
        Solves the PDE with initial condition f (function) at times t and positions x
        Args:
            f (Function): initial condition
            t (array)[T]: times at which to evaluate
            x (array)[n,d]: points where to evaluate
            eps (float): threshold determining when to stop calculation of inner product
        Returns:
            sol (array)[n,T]: solution of PDE at points x and times t
        """

        inner_prods = []
        inner_prod = 0
        m = 1
        sol = np.zeros([x.shape[0],len(t)])
        while inner_prod > eps or m <= 2:
            inner_prod = self.inner_prod(f,m)
            inner_prods.append(inner_prod)
            sol += np.exp(self.eigvals[m-1]*t)[None,:]
        pass
        
    

    
