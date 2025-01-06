import numpy as np
from src.energy.base_energy import BaseEnergy

class GaussianMixture(BaseEnergy):
    """
    Energy function corresponding to a Gaussian mixture
    """

    def __init__(self, weights, means, covs, *args, **kwargs):
        """
        Args:
            weights (ndarray)[M]: weights (should sum to 1)
            means (ndarray)[M,d]: means of the modes
            covs (ndarray)[M,d,d]: cov matrices of modes
        """
        super().__init__(*args, **kwargs)
        self.weights = weights
        self.means = means
        self.covs = covs
        self.M = len(weights)
        self.dim = means.shape[1]
        self.inv_covs = np.linalg.inv(covs)
        self.cov_dets = np.linalg.det(covs)
        self.Ls = np.linalg.cholesky(self.covs)

        
    def forward(self, x):
        """
        Evaluate the energy at the given points.

        E(x) = -log(sum_i w_i p_i(x))
        where p_i(x) = 1/|cov_i|^(1/2)*exp(-1/2 (x-mu_i)T inv_cov_i (x-mu_i))

        Args:
            x (ndarray)[N, d]: points to evaluate at
        Returns:
            energy (ndarray)[N]: energy evaluated at points
        """
        # Energy E(x) = -log(\sum_i w_i p_i(x))
        quadratic_forms = np.einsum('kij,kjj,kij->ik',x[None,:,:]-self.means[:,None,:],self.inv_covs,x[None,:,:]-self.means[:,None,:])
        energy = -np.log(np.sum(self.weights/np.sqrt(self.cov_dets) * np.exp(-1/2 * quadratic_forms),axis=1))
        return energy
    
    def grad(self, x):
        """
        Evaluate the gradient of energy at the given points.

        grad E(x) = -1/(sum_i w_i grad p_i(x))  * (sum_i w_i grad p_i(x))
                  = 1/(sum_i w_i grad p_i(x)) * (sum_i w_i p_i(x) inv_cov_i (x-mu_i))
        Args:
            x (ndarray)[N, d]: points to evaluate
        Returns:
            grad_x (ndarray)[N, d]: gradient of energy evaluated at points
        """
        # N x M
        quadratic_forms = np.einsum('kij,kjj,kij->ik',x[None,:,:]-self.means[:,None,:],self.inv_covs,x[None,:,:]-self.means[:,None,:])
        
        # N x M x D
        matrix_prods = np.einsum('kij, kjj -> ikj', x[None,:,:]-self.means[:,None,:], self.inv_covs)
        
        # N x M
        densities = self.weights/np.sqrt(self.cov_dets) * np.exp(-1/2 * quadratic_forms)
        denom = np.sum(self.weights[None,:,None] * densities[:,:,None], axis=1)

        grad_x = 1/denom * np.sum(self.weights[None,:,None] * densities[:,:,None] * matrix_prods,axis=1)

        return grad_x

    def exact_sample(self, n):
        """
        Compute exact samples from the stationary measure (mixture of multivariate gaussians).

        Args:
            n (tuple): shape of sample
        Returns:
            samples (ndarray)[n, d]: samples
        """
        # Sample from a multivariate normal distribution with mean 0 and covariance matrix inv(A)
        indices = np.random.choice(np.arange(self.M),size = n, p = self.weights)
        
        samples = np.random.standard_normal(n + (self.dim,))

        transformed_samples = (
        np.einsum('nij,nj->ni', self.Ls[indices], samples)  # Scale by Cholesky factors
        + self.means[indices]  # Shift by means
        )

        return transformed_samples
