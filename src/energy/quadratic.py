import torch

from src.energy.base_energy import BaseEnergy

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
        self.L = torch.linalg.cholesky(self.A)

    def forward(self, x):
        """
        Evaluate the energy at the given points.

        Args:
            x (Tensor)[N, d]: points to evaluate at
        Returns:
            energy (Tensor)[N]: energy evaluated at points
        """
        # Energy E(x) = 0.5 * x^T A x
        energy = 0.5 * torch.sum(x @ self.A * x, dim=-1)
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
        sample = torch.randn(n + (self.dim,)) @ self.L.T  
        return sample