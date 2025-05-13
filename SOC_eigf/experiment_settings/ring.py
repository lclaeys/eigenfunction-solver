"""

Model set up for ring setting.

"""
import torch

from SOC_eigf_old2.method import NeuralSDE

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

    def nabla_b(self, t, x):
        x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
        
        I = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
        if len(x.shape) == 2:
            I = I.unsqueeze(0).expand(x.shape[0], -1, -1)
        elif len(x.shape) == 3:
            I = I.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
        
        outer = torch.einsum('...i,...j->...ij', x, x)
                
        a = 2 * x_norm_sq - self.radius**2
        
        jacobian = -2 * self.scale * (4 * outer + a[..., None, None] * I)
        
        return jacobian
    
    def f(self,t, x):
        return torch.einsum("j,...j->...", self.P, x)

    def nabla_f(self,t, x):
        if len(x.shape) == 2:
            return self.P.unsqueeze(0).repeat(x.shape[0], 1)
        elif len(x.shape) == 3:
            return self.P.unsqueeze(0).unsqueeze(0).repeat(
                x.shape[0], x.shape[1], 1
            )
        elif len(x.shape) == 3:
            return self.P.unsqueeze(0).unsqueeze(0).unsqueeze(
                0
            ).repeat(x.shape[0], x.shape[1], x.shape[2], 1)

    # Final cost
    def g(self, x):
        return torch.sum(
            x * torch.einsum("ij,...j->...i", self.Q, x), -1
        )

    # Gradient of final cost
    def nabla_g(self, x):
        return 2 * torch.einsum("ij,...j->...i", self.Q, x)
    
