import numpy as np
import heapq
from itertools import combinations

from src.energy.base_energy import BaseEnergy
from scipy.special import eval_hermitenorm
from scipy.special import factorial

class QuadraticEnergy(BaseEnergy):
    """
    Quadratic energy function with a positive semi-definite matrix A:
        E(x) = 0.5 * x^T A x
    """

    def __init__(self, A, *args, **kwargs):
        """
        Args:
            A (ndarray): Positive semi-definite matrix (d, d)
        """
        super().__init__(*args, **kwargs)
        if not np.all(np.linalg.eigvals(A) >= 0):
            raise ValueError("Matrix A is not positive semi-definite")
        self.A = A
        self.inv_A = np.linalg.inv(self.A)
        self.dim = self.A.shape[0]
        self.compute_indices = 0
        self.diag_A = np.diag(A)

    def forward(self, x):
        """
        Evaluate the energy at the given points.

        Args:
            x (ndarray)[N, d]: points to evaluate at
        Returns:
            energy (ndarray)[N]: energy evaluated at points
        """
        # Energy E(x) = 0.5 * x^T A x
        energy = 0.5 * np.sum(x @ self.A * x, axis=-1)
        return energy
    
    def grad(self, x):
        """
        Evaluate the gradient of energy at the given points.

        Args:
            x (ndarray)[N, d]: points to evaluate
        Returns:
            grad_x (ndarray)[N, d]: gradient of energy evaluated at points
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
            sample (ndarray)[n, d]: samples
        """
        # Sample from a multivariate normal distribution with mean 0 and covariance matrix inv(A)
        sample = np.random.multivariate_normal(np.zeros(self.dim), self.inv_A, size = n)
        return sample
    
    def exact_eigvals(self, m):
        """
        Compute m smallest exact eigenvalues.
        If A = diag(a_1, ..., a_d), then each eigenvalue can be associated with a tuple
        (n_1, ..., n_d), and the corresponding eigenvalue is n_1 a_1 + ... + n_d a_d
        Args:
            m (Int)
        Returns:
            eigvals (ndarray)
        """
        if self.compute_indices != m:
            self._compute_indices(m)

        return self.eigvals

    def exact_eigfunctions(self, x, m):
        """
        Evaluate first m exact eigenfunctions at points x, assuming A is diagonal
        Args:
            x (ndarray)[n,d]: evaluation points
        Returns:
            fx (ndarray)[n,m]: first m eigenfunction evaluations
        """
        if not np.allclose(self.A, np.diag(np.diag(self.A))):
            raise ValueError("Matrix A is not diagonal")

        if self.compute_indices != m:
            self._compute_indices(m)

        fx = np.ones([x.shape[0],m])
        for i in range(m):
            hermite_evals = eval_hermitenorm(self.indices[i],x*np.sqrt(self.diag_A)[None,:])
            norms = np.sqrt(factorial(self.indices[i]))

            hermite_evals /= norms

            if len(hermite_evals.shape) != 1:
                hermite_evals = np.prod(hermite_evals,axis=1)
            fx[:,i] *= hermite_evals
        
        return fx

    def _compute_indices(self, m):
        if self.compute_indices == m:
            return self.indices
        
        self.eigvals, self.indices = self.smallest_combinations(np.diag(self.A), m)

    @staticmethod
    def smallest_combinations(x, m):
        """
        Args:
            x (ndarray)[d]: input array of eigenvalues of A
            m (int) 
        Returns:
            vals (ndarray)[m]: smallest linear combinations of eigenvalues
            vecs (ndarray)[m,d]: indices of those combinations
        """
        n = len(x)
        
        heap = []
        visited = set()
        
        # Start with the combination (0, 0, ..., 0)
        initial = (0, [0] * n)  # (S, a_vector)
        heapq.heappush(heap, initial)
        visited.add(tuple([0] * n))
        
        vals = []
        vecs = []
        
        while len(vals) < m:
            S, a_vector = heapq.heappop(heap)
            vals.append(S) 
            vecs.append(a_vector)

            # Generate new combinations
            for i in range(n):
                new_a_vector = a_vector[:]
                new_a_vector[i] += 1
                new_S = S + x[i]
                
                if tuple(new_a_vector) not in visited:
                    heapq.heappush(heap, (new_S, new_a_vector))
                    visited.add(tuple(new_a_vector))
        
        return np.array(vals), np.array(vecs)




