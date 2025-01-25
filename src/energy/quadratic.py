import numpy as np
import torch as torch
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
            A (tensor): Positive semi-definite matrix (d, d)
        """
        super().__init__(*args, **kwargs)

        if not torch.allclose(A, A.T):
            raise ValueError("Matrix A is not symmetric.")
        
        if not torch.all(torch.linalg.eigvalsh(A) >= 0):
            raise ValueError("Matrix A is not positive semi-definite")
        
        self.A = A
        self.dim = self.A.size(0)
        self.compute_indices = 0
        
        self.D, self.U = torch.linalg.eigh(A)

        self.distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.dim), precision_matrix = self.A
        )

    def forward(self, x):
        """
        Evaluate the energy at the given points.

        Args:
            x (tensor)[N, d]: points to evaluate at
        Returns:
            energy (tensor)[N]: energy evaluated at points
        """
        log_prob = self.distribution.log_prob(x)
        energy = -log_prob
        return energy
    
    def grad(self, x):
        """
        Evaluate the gradient of energy at the given points.

        Args:
            x (tensor)[N, d]: points to evaluate
        Returns:
            grad_x (tensor)[N, d]: gradient of energy evaluated at points
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
            sample (tensor)[n, d]: samples
        """
        # Sample from a multivariate normal distribution with mean 0 and covariance matrix inv(A)
        sample = self.distribution.rsample(n)
        return sample
    
    def exact_eigvals(self, m):
        """
        Compute m smallest exact eigenvalues.
        If A = diag(a_1, ..., a_d), then each eigenvalue can be associated with a tuple
        (n_1, ..., n_d), and the corresponding eigenvalue is n_1 a_1 + ... + n_d a_d
        Args:
            m (Int)
        Returns:
            eigvals (tensor)
        """
        if self.compute_indices != m:
            self._compute_indices(m)

        return self.eigvals

    def exact_eigfunctions(self, x, m, use_scipy = True):
        """
        Evaluate first m exact eigenfunctions at points x
        Args:
            x (tensor)[n,d]: evaluation points
            m (int): number of eigenfunctions to compute
            use_scipy (bool): whether to use scipy for evaluating hermite polynomials (not differentiable)
        Returns:
            fx (tensor)[n,m]: first m eigenfunction evaluations
        """
        if x.requires_grad:
            use_scipy = False
        
        if self.compute_indices != m:
            self._compute_indices(m)

        reshaped_x = (x @ self.U) * torch.sqrt(self.D)[None,:]

        if use_scipy:
            fx = np.ones([x.shape[0],m])
            for i in range(m):
                # numpy
                hermite_evals = eval_hermitenorm(self.indices[i],reshaped_x.numpy())
                norms = np.sqrt(factorial(self.indices[i]))

                hermite_evals /= norms

                if len(hermite_evals.shape) != 1:
                    hermite_evals = np.prod(hermite_evals,axis=1)

                fx[:,i] *= hermite_evals
        
            fx = torch.tensor(fx,dtype = x.dtype, device = x.device)

        else:
            fx = torch.ones([x.shape[0],m], dtype = x.dtype, device = x.device)
            for i in range(m):
                # pytorch
                n = torch.tensor(self.indices[i], device = x.device)
                hermite_evals = self.eval_hermitenorm(n,reshaped_x)
                norms = torch.sqrt(torch.tensor(factorial(self.indices[i]), dtype = x.dtype, device = x.device))

                hermite_evals /= norms

                if len(hermite_evals.shape) != 1:
                    hermite_evals = torch.prod(hermite_evals,dim=1)

                fx[:,i] *= hermite_evals
            
        return fx

    def _compute_indices(self, m):
        if self.compute_indices == m:
            return self.indices
        
        self.eigvals, self.indices = self.smallest_combinations(self.D, m)

    def generate_probabilist_hermite_coeffs(self, m, dtype):
        """
        Generate the coefficients of the first m probabilist Hermite polynomials.
        
        Args:
            m (int): Number of Hermite polynomials to generate (non-negative integer).
            dtype
        Returns:
            torch.Tensor: A 2D tensor of shape (m, m), where each row contains the 
                        coefficients of the corresponding Hermite polynomial, 
                        padded with zeros for alignment.
        """
        # Initialize a tensor to store coefficients
        coeffs = torch.zeros((m, m), dtype=dtype)
        
        # H_0(x) = 1
        if m > 0:
            coeffs[0, 0] = 1.0
        
        # H_1(x) = x
        if m > 1:
            coeffs[1, 1] = 1.0
        
        # Use the recurrence relation to compute higher-order coefficients
        for n in range(2, m):
            # H_n(x) = x * H_{n-1}(x) - (n-1) * H_{n-2}(x)
            coeffs[n, 1:] += coeffs[n-1, :-1]  # x * H_{n-1}(x)
            coeffs[n, :] -= (n - 1) * coeffs[n-2, :]  # - (n-1) * H_{n-2}(x)
        
        return coeffs

    def eval_hermitenorm(self, n, x):
        """
        Implementation of scipy's eval_hermitenorm in a differentiable way.
        Args:
            n (tensor)[d]: degree of polynomial to evaluate at each dimension
            x (tensor)[N,d]: points to evaluate
        Returns:
            He (tensor)[N,d]: values of He polynomial
        """
        m = n.max() + 1

        coeff_matrix = self.generate_probabilist_hermite_coeffs(m, dtype = x.dtype)

        # (d,m)
        coeffs = coeff_matrix[n]
        
        x_powers = torch.cumprod(x.repeat(m-1,1,1),0)

        # shape (m, N, d)
        full_x_powers = torch.ones((m,) + x.shape)
        full_x_powers[1:] = x_powers

        return torch.einsum('ijk, ki -> jk', full_x_powers, coeffs)
    
    @staticmethod
    def smallest_combinations(x, m):
        """
        Args:
            x (tensor)[d]: input array of eigenvalues of A
            m (int) 
        Returns:
            vals (tensor)[m]: smallest linear combinations of eigenvalues
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
        
        return torch.tensor(vals), np.array(vecs)




