"""

Model setup for ring particles setting.

"""
import torch

from SOC_eigf.method import NeuralSDE

class RingParticles(NeuralSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        u=None,
        lmbd=1.0,
        sigma=torch.eye(2),
        radius = 1.0,
        scale = 1.0,
        P=torch.ones(2),
        R = 1.0,
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
        self.R = R

        self.confining=True

        assert self.dim % 2 == 0

    # Energy: b = -grad E
    def energy(self, x):
        x_reshaped = x.reshape(x.shape[0],-1,2)
        xnorm = torch.norm(x_reshaped, dim=-1)
        E_terms = self.scale * (xnorm**2 - self.radius**2) * xnorm**2
        return E_terms.sum(dim=1)

    # Base Drift
    def b(self, t, x):
        x_reshaped = x.reshape(x.shape[0],-1,2)
        xnorm = torch.norm(x_reshaped, dim=-1,keepdim=True)
        b = -2 * self.scale * (2 * x_reshaped * xnorm**2 - self.radius**2 * x_reshaped)
        return b.reshape(x.shape)

    # Gradient (Jacobian) of Drift
    def nabla_b(self, t, x):
        # Compute squared norm of x; keep the last dimension for broadcasting.
        raise NotImplementedError
    
    def f(self,t, x):
        x_reshaped = x.reshape(x.shape[0],-1,2)
        external_field_cost = torch.einsum("j,...j->...", self.P, x_reshaped).sum(dim=1)

        squared_dists = torch.cdist(x_reshaped,x_reshaped)**2
        interaction_cost = squared_dists.sum(dim=[1, 2]) / 2

        return external_field_cost + self.R * interaction_cost

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
        return torch.zeros(x.shape[0],device=x.device)

    # Gradient of final cost
    def nabla_g(self, x):
        return 2 * torch.einsum("ij,...j->...i", self.Q, x)
    
