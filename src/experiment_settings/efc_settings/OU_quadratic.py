# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import torch

from src.soc.method import EigenSDE

class OU_Quadratic(EigenSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        hdims=[256, 128, 64],
        prior=None,
        joint=True,
        u=None,
        lmbd=1.0,
        A=torch.eye(2),
        P=torch.eye(2),
        Q=torch.eye(2),
        sigma=torch.eye(2),
        T = 1.0,
        k = 1,
    ):
        super().__init__(
            device=device,
            dim=dim,
            hdims=hdims,
            u=u,
            lmbd=lmbd,
            sigma=sigma,
            T=T,
            k=k,
        )
        self.A = A
        self.P = P
        self.Q = Q
        self.prior = prior
        self.joint = joint
        
        eigvals = torch.linalg.eigvalsh(A)
        if torch.all(eigvals > 0):
            self.confining = False
        elif torch.all(eigvals < 0):
            self.confining = True
        else:
            print('Error: A is neither confining nor repulsive.')
            raise NotImplementedError


    # Energy: b = -grad E
    def energy(self, x):
        return -0.5 * torch.sum(
            x * torch.einsum("ij,...j->...i", self.A, x), -1
        )

    # Base Drift
    def b(self, t, x):
        return torch.einsum("ij,...j->...i", self.A, x)

    # Gradient of base drift
    def nabla_b(self, t, x):
        if len(x.shape) == 2:
            return torch.transpose(self.A.unsqueeze(0).repeat(x.shape[0], 1, 1), 1, 2)
        elif len(x.shape) == 3:
            return torch.transpose(
                self.A.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1, 1),
                2,
                3,
            )

    # Running cost
    def f(self, t, x):
        return torch.sum(
            x * torch.einsum("ij,...j->...i", self.P, x), -1
        )

    # Gradient of running cost
    def nabla_f(self, t, x):
        return 2 * torch.einsum("ij,...j->...i", self.P, x)

    # Final cost
    def g(self, x):
        return torch.sum(
            x * torch.einsum("ij,...j->...i", self.Q, x), -1
        )

    # Gradient of final cost
    def nabla_g(self, x):
        return 2 * torch.einsum("ij,...j->...i", self.Q, x)
    
