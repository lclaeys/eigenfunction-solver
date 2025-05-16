"""
Model setup for a *ring* potential (a quartic “Mexican-hat”) that confines
trajectories to a sphere of radius ``radius``.  Written to slot straight into
the SOC-EigF framework just like `OU_Quadratic` above, but **shape-agnostic**:
all state-tensors may have any leading axes, e.g. (time, batch, d) or
(batch, time, d).

E(x) = scale · (‖x‖² − radius²) · ‖x‖²                        (1)

The base drift is minus its gradient and grows as **O(r³)** away from the ring,
so it is strongly confining.

Author: ChatGPT  (May 2025)
"""

from __future__ import annotations

import torch
from SOC_eigf.method import NeuralSDE


class Ring(NeuralSDE):
    # --------------------------------------------------------------------- init
    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dim: int = 2,
        u=None,
        lmbd: float = 1.0,
        sigma: torch.Tensor | None = None,
        radius: float = 1.0,
        scale: float = 1.0,
        P: torch.Tensor | None = None,      # vector (running cost)
        Q: torch.Tensor | None = None,      # matrix (terminal cost)
        T: float = 1.0,
        method: str = "EIGF",
        eigf_cfg=None,
        ido_cfg=None,
    ):
        device = torch.device(device)
        dtype = torch.float32

        # Safe defaults ---------------------------------------------------------
        if sigma is None:
            sigma = torch.eye(dim, dtype=dtype, device=device)
        else:
            sigma = sigma.to(device, dtype)

        if P is None:
            P = torch.ones(dim, dtype=dtype, device=device)
        else:
            if P.ndim != 1 or P.shape[0] != dim:
                raise ValueError("P must be a 1-D tensor of length `dim`")
            P = P.to(device, dtype)

        if Q is None:
            Q = torch.eye(dim, dtype=dtype, device=device)
        else:
            if Q.shape != (dim, dim):
                raise ValueError("Q must have shape (dim, dim)")
            Q = Q.to(device, dtype)

        super().__init__(
            device=device,
            dim=dim,
            u=u,
            lmbd=lmbd,
            sigma=sigma,
            T=T,
            method=method,
            eigf_cfg=eigf_cfg,
            ido_cfg=ido_cfg,
        )

        self.radius = float(radius)
        self.scale = float(scale)
        self.P = P
        self.Q = Q

        self.confining = True  # used by higher-level code

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _identity_like(x: torch.Tensor) -> torch.Tensor:
        """Return an identity matrix broadcast to match x with shape (..., d)."""
        d = x.shape[-1]
        I = torch.eye(d, device=x.device, dtype=x.dtype)
        I = I.reshape(*((1,) * (x.ndim - 1)), d, d)
        return I.expand(*x.shape[:-1], d, d)

    # ------------------------------------------------------------------ physics
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        r2 = torch.sum(x**2, dim=-1)
        return self.scale * (r2 - self.radius**2) * r2

    # Drift  b = −∇E
    def b(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r2 = torch.sum(x**2, dim=-1, keepdim=True)
        return -2.0 * self.scale * (2.0 * x * r2 - self.radius**2 * x)

    # Jacobian ∇ₓ b
    def nabla_b(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r2 = torch.sum(x**2, dim=-1, keepdim=False)            # (..., 1)
        outer = torch.einsum("...i,...j->...ij", x, x)        # (..., d, d)
        a = 2.0 * r2 - self.radius**2                         # (..., 1)
        I = self._identity_like(x)                            # (..., d, d)
        return -2.0 * self.scale * (4.0 * outer + a[..., None, None] * I)

    # ------------------------------------------------------------------- costs
    # Running cost  f = P·x  (scalar)
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("i,...i->...", self.P, x)

    # ∇ₓ f = P  (broadcasted over leading axes)
    def nabla_f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.P.expand(*x.shape[:-1], -1)

    # Terminal cost  g = xᵀ Q x
    def g(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i,ij,...j->...", x, self.Q, x)

    # ∇ₓ g = 2 Q x
    def nabla_g(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.einsum("ij,...j->...i", self.Q, x)