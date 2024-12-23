import numpy as np
from itertools import combinations

from src.energy.base_energy import BaseEnergy
from scipy.special import eval_hermitenorm

class QuadraticEnergy(BaseEnergy):
    """
    Quadratic energy function with a positive semi-definite matrix A:
        E(x) = 0.5 * x^T A x
    """

    def __init__(self, A, *args, **kwargs):
        """
        Args:
            A (Tensor): Positive semi-definite matrix (d, d)
        """
        super().__init__(*args, **kwargs)
        self.A = A
        self.dim = self.A.shape[0]
        self.L = np.linalg.cholesky(self.A)
        self.compute_indices = 0
    def forward(self, x):
        """
        Evaluate the energy at the given points.

        Args:
            x (Tensor)[N, d]: points to evaluate at
        Returns:
            energy (Tensor)[N]: energy evaluated at points
        """
        # Energy E(x) = 0.5 * x^T A x
        energy = 0.5 * np.sum(x @ self.A * x, axis=-1)
        return energy
    
    def grad(self, x):
        """
        Evaluate the gradient of energy at the given points.

        Args:
            x (Tensor)[N, d]: points to evaluate
        Returns:
            grad_x (Tensor)[N, d]: gradient of energy evaluated at points
        """
        # Gradient of 0.5 * x^T A x is A x
        grad_x = x @ self.A
        return grad_x
    
    def exact_sample(self, n):
        """
        Compute exact samples from the stationary measure (multivariate normal).

        Args:
            n (tuple): shape of sample
        Returns:
            sample (Tensor)[n, d]: samples
        """
        # Sample from a multivariate normal distribution with mean 0 and covariance matrix A
        sample = np.random.standard_normal(n + (self.dim,)) @ self.L.T  
        return sample
    
    def exact_eigvals(self, m):
        if self.compute_indices != m:
            self._compute_indices(m)

        return self.indices.sum(axis=1) 

    def exact_eigfunctions(self, x, m):
        """
        Evaluate first m exact eigenfunctions at points x, assuming A = I
        Args:
            x (array)[n,d]: evaluation points
        Returns:
            fx (array)[n,m]: first m eigenfunction evaluations
        """
        if self.compute_indices != m:
            self._compute_indices(m)

        fx = np.ones([x.shape[0],m])
        for i in range(m):
            hermite_evals = eval_hermitenorm(self.indices[i],x)
            if len(hermite_evals.shape) != 1:
                hermite_evals = np.prod(hermite_evals,axis=1)
            fx[:,i] *= hermite_evals
        
        return fx

    def _compute_indices(self, m):
        if self.compute_indices == m:
            return self.indices
        
        i = 1
        self.indices = np.zeros([m,self.dim],dtype=int)
        eigval = 1
        while i < m:
            combinations = self._generate_combinations(eigval,self.dim)
            j = min(combinations.shape[0],m-i)
            self.indices[i:i+j,:] = combinations[:j]
            eigval += 1
            i += j
        
        self.compute_indices = m

    @staticmethod
    def _generate_combinations(k, n):
        # Total number of slots (k objects + n-1 dividers)
        total_slots = k + n - 1
        # Indices for dividers (choose n-1 positions for dividers from total slots)
        divider_positions = list(combinations(range(total_slots), n - 1))
        # Generate combinations
        all_combinations = []
        for dividers in divider_positions:
            combination = np.zeros(n, dtype=int)
            start = 0
            for i, divider in enumerate(dividers):
                combination[i] = divider - start
                start = divider + 1
            combination[-1] = total_slots - start  # Remaining objects in the last box
            all_combinations.append(combination)
        return np.array(all_combinations)


        

