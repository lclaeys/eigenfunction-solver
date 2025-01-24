import torch
import torch.nn as nn
import torch.distributions as dist
from src.energy.base_energy import BaseEnergy

class GaussianMixture(BaseEnergy):
    """
    Energy function corresponding to a Gaussian mixture
    """

    def __init__(self, weights, means, covs):
        """
        Args:
            weights (Tensor)[M]: weights (should sum to 1)
            means (Tensor)[M,d]: means of the modes
            covs (Tensor)[M,d,d]: cov matrices of modes
        """
        super(GaussianMixture, self).__init__()
        self.weights = weights
        self.means = means
        self.covs = covs
        self.M = weights.size(0)
        self.dim = means.size(1)
        self.mixture = dist.MixtureSameFamily(
            dist.Categorical(self.weights),
            dist.MultivariateNormal(self.means, covariance_matrix=self.covs)
        )

    def forward(self, x):
        """
        Evaluate the energy at the given points.

        E(x) = -log(sum_i w_i p_i(x))
        where p_i(x) = 1/|cov_i|^(1/2)*exp(-1/2 (x-mu_i)T inv_cov_i (x-mu_i))

        Args:
            x (Tensor)[N, d]: points to evaluate at
        Returns:
            energy (Tensor)[N]: energy evaluated at points
        """
        log_prob = self.mixture.log_prob(x)
        energy = -log_prob
        return energy

    def grad(self, x):
        """
        Compute the gradient of the energy at the given points.

        Args:
            x (Tensor)[N, d]: points to evaluate at
        Returns:
            grad (Tensor)[N, d]: gradient of the energy at points
        """
        if not x.requires_grad:
            x.requires_grad_(True)
        energy = self.forward(x)

        # Here is how backward works. Let us vectorize input x and output y=f(x)
        # Then, taking gradient of y w.r.t. x gives us Jacobian matrix J_f
        # J_f[i,j] = d y_i / d x_j
        # The backward operation in pytorch returns a value of v^T J_f, where v has to be specified in a non-scalar case
        # Note that if x is a batch (ie input rows are evaluated independently), 
        # then putting v = (1,1,..., 1) gives us the desired result, after reshaping v^T J_f to the shape of x
        # If it is not independent, then in general we can not do that, and we would need to restore whole Jacobian by multiple queries.
        # Note that it is also similar to applying backward to (1, 1, ..., 1)^T y

        energy.backward(gradient=torch.ones_like(energy),retain_graph=True) 
        grad = x.grad.detach()
        x.requires_grad_(False)

        return grad
    
    # def grad(self, x):
    #     """
    #     Compute the gradient of the energy at the given points.

    #     Args:
    #         x (Tensor)[N, d]: points to evaluate at
    #     Returns:
    #         grad (Tensor)[N, d]: gradient of the energy at points
    #     """
    #     if not x.requires_grad:
    #         x.requires_grad_(True)
        
    #     energy = self.forward(x)
        
    #     # Use autograd.grad for explicit gradient computation
    #     grad = torch.autograd.grad(
    #         outputs=energy,
    #         inputs=x,
    #         grad_outputs=torch.ones_like(energy),
    #     )[0]
        
    #     return grad

    def exact_sample(self, num_samples):
        """
        Sample from the Gaussian mixture distribution.

        Args:
            num_samples (tuple): shape of samples to generate
        Returns:
            samples (Tensor)[num_samples, d]: generated samples
        """
        samples = self.mixture.sample(num_samples)
        return samples