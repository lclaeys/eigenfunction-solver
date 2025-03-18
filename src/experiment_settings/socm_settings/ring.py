from SOC_matching import method
import torch

class Ring(method.NeuralSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        hdims=[256, 128, 64],
        hdims_M=[128, 128],
        u=None,
        lmbd=1.0,
        radius=1.0,
        scale=1.0,
        P=torch.eye(2),
        Q=torch.eye(2),
        sigma=torch.eye(2),
        gamma=3.0,
        scaling_factor_nabla_V=1.0,
        scaling_factor_M=1.0,
        T=1.0,
        u_warm_start=None,
        use_warm_start=False,
    ):
        super().__init__(
            device=device,
            dim=dim,
            hdims=hdims,
            hdims_M=hdims_M,
            u=u,
            lmbd=lmbd,
            sigma=sigma,
            gamma=gamma,
            scaling_factor_nabla_V=scaling_factor_nabla_V,
            scaling_factor_M=scaling_factor_M,
            T=T,
            u_warm_start=u_warm_start,
            use_warm_start=use_warm_start,
        )
        self.radius = radius
        self.scale = scale
        self.P = P
        self.Q = Q


    def energy(self, x):
        xnorm = torch.norm(x, dim=-1)
        return (xnorm**2 - self.radius**2) * xnorm**2

    # Base Drift
    def b(self, t, x):
        xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
        return -2 * self.scale * (2 * x * xnorm**2 - self.radius**2 * x)

    # Gradient (Jacobian) of Drift
    def nabla_b(self, t, x):
        # Compute squared norm of x; keep the last dimension for broadcasting.
        x_norm_sq = torch.sum(x**2, dim=-1, keepdim=False)
        
        # Create an identity matrix of size equal to the last dimension of x.
        I = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
        # Adjust shape of I based on the dimensionality of x.
        if len(x.shape) == 2:
            # x has shape (batch, d) --> I becomes (batch, d, d)
            I = I.unsqueeze(0).expand(x.shape[0], -1, -1)
        elif len(x.shape) == 3:
            # x has shape (batch, N, d) --> I becomes (batch, N, d, d)
            I = I.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
        
        # Compute the outer product of x with itself.
        outer = torch.einsum('...i,...j->...ij', x, x)
        
        # Compute the factor a(x) = 2*||x||^2 - self.radius**2.
        # Its shape is broadcastable to the last two dims.
        a = 2 * x_norm_sq - self.radius**2
        
        # Combine terms to compute the Jacobian:
        # nabla_b(x) = 2 * self.scale * (4 * outer + a[..., None, None] * I)
        jacobian = - 2 * self.scale * (4 * outer + a[..., None, None] * I)
        
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
