"""

Model set up for ring setting. IDO method not supported.

"""
import torch

from SOC_eigf.method import NeuralSDE

class Ring(NeuralSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        u=None,
        lmbd=1.0,
        sigma=torch.eye(2),
        radius = 1.0,
        scale = 1.0,
        P=torch.eye(2),
        Q=torch.eye(2),
        T = 1.0,
        method = "EIGF",
        eigf_cfg=None,
        ido_cfg=None
    ):
        super().__init__(
            device=device,
            dim=dim,
            u=u,
            lmbd=lmbd,
            sigma=sigma,
            T=T,
            method=method,
            eigf_cfg=eigf_cfg,
            ido_cfg=ido_cfg
        )
        self.radius = radius
        self.scale = scale
        self.P = P
        self.Q = Q

        self.confining=True

    # Energy: b = -grad E
    def energy(self, x):
        xnorm = torch.norm(x, dim=-1)
        return self.scale * (xnorm**2 - self.radius**2) * xnorm**2

    # Base Drift
    def b(self, t, x):
        xnorm = torch.norm(x, dim=-1).unsqueeze(-1)

        return -2 * self.scale * (2 * x * xnorm**2 - self.radius**2 * x)

    def f(self,t, x):
        return torch.einsum("j,...j->...", self.P, x)

    # Final cost
    def g(self, x):
        return torch.sum(
            x * torch.einsum("ij,...j->...i", self.Q, x), -1
        )