import torch as torch

from src.energy.base_energy import BaseEnergy


class RingEnergy(BaseEnergy):
    """
    "Ring energy", aka "Zebang's favourite function", given by
    
    E(x) = scale * (1/3 * |x|^3 - r/2 |x|^2)
    
    The set of minimizers is the sphere centered at the origin with radius r.
    """

    def __init__(self, dim, scale, radius, *args, **kwargs):
        """
        Args:
            dim (int): dimension
            scale (float): scale
            radius (float): radius
        """
        super(RingEnergy, self).__init__(*args, **kwargs)
        self.dim = dim
        self.scale = scale
        self.radius = radius

    def forward(self, x):
        """
        Evaluate the energy at the given points.

        Args:
            x (tensor)[N, d]: points to evaluate at
        Returns:
            energy (tensor)[N]: energy evaluated at points
        """
        norm = x.norm(dim=1)
        energy = self.scale * (norm ** 2) * (norm/3 - self.radius/2)
        return energy
    
    def grad(self, x):
        """
        Evaluate the gradient of energy at the given points.

        Args:
            x (tensor)[N, d]: points to evaluate
        Returns:
            grad_x (tensor)[N, d]: gradient of energy evaluated at points
        """
        # Gradient of |x|^n is nx|x|^{n-2}
        norm = x.norm(dim = 1)
        grad_x = self.scale * x *(norm[:,None] - self.radius)
        return grad_x 