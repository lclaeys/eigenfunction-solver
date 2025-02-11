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
        super(QuadraticEnergy, self).__init__(*args, **kwargs)

        if not torch.allclose(A, A.T):
            raise ValueError("Matrix A is not symmetric.")
        
        confine_mult = 1.0
        if not torch.all(torch.linalg.eigvalsh(A) >= 0):
            print('Warning: matrix A is not positive definite. Using -A for samples, but some functions may not work as intended.')
            self.non_confining = True
            confine_mult = -1.0
        
        self.A = A
        self.dim = self.A.size(0)
        self.compute_indices = 0
        
        self.D, self.U = torch.linalg.eigh(A)

        self.distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.dim,device=A.device), precision_matrix = self.A * confine_mult
        )

        self.stored_coeff_matrix = torch.ones((1,1),device=A.device)
        self.stored_grad_coeff_matrix = torch.zeros((1,1),device=A.device)

    def forward(self, x):
        """
        Evaluate the energy at the given points.

        Args:
            x (tensor)[N, d]: points to evaluate at
        Returns:
            energy (tensor)[N]: energy evaluated at points
        """
        energy = 1/2 * torch.einsum("ij, ij -> i", x @ self.A, x)
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

    def exact_eigfunctions(self, x, m, use_scipy = True, return_grad = False):
        """
        Evaluate first m exact eigenfunctions at points x
        Args:
            x (tensor)[n,d]: evaluation points
            m (int): number of eigenfunctions to compute
            use_scipy (bool): whether to use scipy for evaluating hermite polynomials (not differentiable)
            return_grad (bool): whether to return gradient as well
        Returns:
            fx (tensor)[n,m]: first m eigenfunction evaluations
            (Optinal) grad_fx [n,m,d]: gradients of first m eigenfunctions
        """
        if x.requires_grad or return_grad:
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
                    fx[:,i] = np.prod(hermite_evals,axis=1)
        
            fx = torch.tensor(fx,dtype = x.dtype, device = x.device)
        else:
            fx = torch.ones([x.shape[0],m], dtype = x.dtype, device = x.device)
            grad_fx = torch.ones([x.shape[0],m,x.shape[1]], dtype = x.dtype, device = x.device)

            for i in range(m):
                # pytorch
                n = torch.tensor(self.indices[i], device = 'cpu') #indices on cpu
                if return_grad:
                    hermite_evals, grad_hermite_evals = self.eval_hermitenorm(n,reshaped_x, return_grad=True)
                else:
                    hermite_evals = self.eval_hermitenorm(n,reshaped_x, return_grad=False)

                norms = torch.sqrt(torch.tensor(factorial(self.indices[i]), dtype = x.dtype, device = x.device))

                hermite_evals /= norms

                fx[:,i]  = torch.prod(hermite_evals,dim=1)

                if return_grad:
                    grad_hermite_evals /= norms

                    # (n,d)
                    ratio_term = grad_hermite_evals / hermite_evals
                    grad_fx[:,i,:] = fx[:,i,None] * (ratio_term * torch.sqrt(self.D)[None,:]) @ self.U.T
        
        if return_grad:
            return fx, grad_fx
        return fx

    def _compute_indices(self, m):
        if self.compute_indices == m:
            return self.indices
        
        self.eigvals, self.indices = self.smallest_combinations(self.D, m)

    def generate_probabilist_hermite_coeffs(self, m, dtype, device):
        """
        Generate the coefficients of the first m probabilist Hermite polynomials.
        
        Args:
            m (int): Number of Hermite polynomials to generate (non-negative integer).
            dtype
            device
        Returns:
            torch.Tensor: A 2D tensor of shape (m, m), where each row contains the 
                        coefficients of the corresponding Hermite polynomial, 
                        padded with zeros for alignment.
        """
        # Initialize a tensor to store coefficients
        coeffs = torch.zeros((m, m), dtype=dtype, device = device)
        
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

    def eval_hermitenorm(self, n, x, return_grad=False):
        """
        Implementation of scipy's eval_hermitenorm in a differentiable way using torch.pow.
        Args:
            n (tensor)[d]: degree of polynomial to evaluate at each dimension
            x (tensor)[N,d]: points to evaluate
            return_grad (bool): whether to return gradient as well
        Returns:
            He (tensor)[N,d]: values of He polynomial
            (Optional) grad_He (tensor)[N,d]: derivative of polynomial
        """
        m = n.max() + 1  # Maximum degree + 1

        # Generate probabilist Hermite coefficients
        if self.stored_coeff_matrix.shape[0] < m:
            self.stored_coeff_matrix = self.generate_probabilist_hermite_coeffs(m, dtype=x.dtype, device=x.device)
            self.stored_grad_coeff_matrix = torch.roll(self.stored_coeff_matrix,-1,1)
            self.stored_grad_coeff_matrix[:,-1] = 0
            self.stored_grad_coeff_matrix = self.stored_grad_coeff_matrix * torch.arange(1,m+1, device = x.device)[None,:]

        # (d, m): Extract the relevant coefficients for the input degrees
        coeff_matrix = self.stored_coeff_matrix[:m,:m]
        grad_coeff_matrix =  self.stored_grad_coeff_matrix[:m,:m]
        coeffs = coeff_matrix[n]
        grad_coeffs = grad_coeff_matrix[n]

        # Compute powers of x using torch.pow
        x_powers = torch.stack([torch.pow(x, i) for i in range(m)], dim=0)  # Shape: (m, N, d)

        # Compute Hermite polynomial values using einsum
        if return_grad:
            return torch.einsum('ijk,ki->jk', x_powers, coeffs), torch.einsum('ijk,ki->jk', x_powers, grad_coeffs)
        
        return torch.einsum('ijk,ki->jk', x_powers, coeffs)

    # def eval_hermitenorm(self, n, x):
    #     """
    #     Implementation of scipy's eval_hermitenorm in a differentiable way.
    #     Args:
    #         n (tensor)[d]: degree of polynomial to evaluate at each dimension
    #         x (tensor)[N,d]: points to evaluate
    #     Returns:
    #         He (tensor)[N,d]: values of He polynomial
    #     """
    #     m = n.max() + 1

    #     coeff_matrix = self.generate_probabilist_hermite_coeffs(m, dtype = x.dtype, device= x.device)

    #     # (d,m)
    #     coeffs = coeff_matrix[n]
        
    #     x_powers = torch.cumprod(x.repeat(m-1,1,1),0)

    #     # shape (m, N, d)
    #     full_x_powers = torch.ones((m,) + x.shape, device = x.device)
    #     full_x_powers[1:] = x_powers

    #     return torch.einsum('ijk, ki -> jk', full_x_powers, coeffs)
    
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




