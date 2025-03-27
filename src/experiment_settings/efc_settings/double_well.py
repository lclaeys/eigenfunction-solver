# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import numpy as np
import torch
from scipy.linalg import solve_banded

from src.soc.method import EigenSDE

class DoubleWell(EigenSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        hdims=[256, 128, 64],
        prior=None,
        joint=True,
        u=None,
        lmbd=1.0,
        kappa=torch.ones(2),
        nu=torch.ones(2),
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
            prior=prior,
            joint=joint
        )
        self.kappa = kappa
        self.nu = nu
        self.confining=True
    
    # Energy: b = -grad E
    def energy(self, x):
        if len(x.shape) == 2:
            return torch.sum(self.kappa.unsqueeze(0) * (x**2 - 1) ** 2, dim = -1)
        elif len(x.shape) == 3:
            return torch.sum(self.kappa.unsqueeze(0).unsqueeze(0) * (x**2 - 1) ** 2, dim = -1)
        
    # Base Drift
    def b(self, t, x):
        if len(x.shape) == 2:
            return -2 * self.kappa.unsqueeze(0) * (x**2 - 1) * 2 * x
        elif len(x.shape) == 3:
            return -2 * self.kappa.unsqueeze(0).unsqueeze(0) * (x**2 - 1) * 2 * x

    # Gradient of base drift
    def nabla_b(self, t, x):
        if len(x.shape) == 2:
            return -torch.diag_embed(
                8 * self.kappa.unsqueeze(0) * x**2
                + 4 * self.kappa.unsqueeze(0) * (x**2 - 1)
            )
        elif len(x.shape) == 3:
            return -torch.diag_embed(
                8 * self.kappa.unsqueeze(0).unsqueeze(0) * x**2
                + 4 * self.kappa.unsqueeze(0).unsqueeze(0) * (x**2 - 1)
            )

    # Running cost
    def f(self, t, x):
        if len(x.shape) == 2:
            return torch.sum(
                self.nu.unsqueeze(0) * (x**2 - 1) ** 2, dim=1
            )
        elif len(x.shape) == 3:
            return torch.sum(
                self.nu.unsqueeze(0).unsqueeze(0) * (x**2 - 1) ** 2,
                dim=2,
            )

    # Gradient of running cost
    def nabla_f(self, t, x):
        if len(x.shape) == 2:
            return 2 * self.nu.unsqueeze(0) * (x**2 - 1) * 2 * x
        elif len(x.shape) == 3:
            return (
                2
                * self.nu.unsqueeze(0).unsqueeze(0)
                * (x**2 - 1)
                * 2
                * x
            )

    # Final cost
    def g(self, x):
        if len(x.shape) == 2:
            return torch.sum(
                self.nu.unsqueeze(0) * (x**2 - 1) ** 2, dim=1
            )
        elif len(x.shape) == 3:
            return torch.sum(
                self.nu.unsqueeze(0).unsqueeze(0) * (x**2 - 1) ** 2,
                dim=2,
            )

    # Gradient of final cost
    def nabla_g(self, x):
        if len(x.shape) == 2:
            return 2 * self.nu.unsqueeze(0) * (x**2 - 1) * 2 * x
        elif len(x.shape) == 3:
            return (
                2
                * self.nu.unsqueeze(0).unsqueeze(0)
                * (x**2 - 1)
                * 2
                * x
            )

    # Potential
    def potential(self, x):
        # return torch.einsum('j,bj->b', self.gamma, x)
        if len(x.shape) == 2:
            return torch.sum(
                self.kappa.unsqueeze(0) * (x**2 - 1) ** 2, dim=1
            )
        elif len(x.shape) == 1:
            return torch.sum(self.kappa * (x**2 - 1) ** 2)

    # Scalar potential
    def scalar_potential(self, x, idx, cpu=False):
        if cpu:
            return self.kappa.cpu()[idx] * (x**2 - 1) ** 2
        else:
            return self.kappa[idx] * (x**2 - 1) ** 2

    # Scalar Base Drift
    def scalar_b(self, t, x, idx):
        return -2 * self.kappa[idx] * (x**2 - 1) * 2 * x

    # Running cost
    def scalar_f(self, t, x, idx,cpu=False):
        if cpu:
            return self.nu.cpu()[idx] * (x**2 - 1) ** 2
        else:
            return self.nu[idx] * (x**2 - 1) ** 2

    # Final cost
    def scalar_g(self, x, idx, cpu=False):
        if cpu:
            return self.nu.cpu()[idx] * (x**2 - 1) ** 2
        else:
            return self.nu[idx] * (x**2 - 1) ** 2

    # Optimal control with running cost
    def compute_reference_solution(
        self, T=1.0, delta_t=0.005, delta_x=0.005, xb=2.5, lmbd=1.0, idx=0
    ):
        nx = int(2.0 * xb / delta_x)
        beta = 2
        xvec = np.linspace(-xb, xb, nx, endpoint=True)

        # Build the finite-difference matrix A for the operator L.
        # (Assumes Neumann boundary conditions.)
        A = np.zeros([nx, nx])
        for i in range(0, nx):
            x = -xb + (i + 0.5) * delta_x
            if i > 0:
                x0 = -xb + (i - 0.5) * delta_x
                x1 = -xb + i * delta_x
                A[i, i - 1] = (
                    -np.exp(
                        beta
                        * 0.5
                        * (
                            self.scalar_potential(x0, idx, cpu=True)
                            + self.scalar_potential(x, idx, cpu=True)
                            - 2 * self.scalar_potential(x1, idx, cpu=True)
                        )
                    )
                    / delta_x**2
                )
                A[i, i] = (
                    np.exp(
                        beta
                        * (
                            self.scalar_potential(x, idx, cpu=True)
                            - self.scalar_potential(x1, idx, cpu=True)
                        )
                    )
                    / delta_x**2
                )
            if i < nx - 1:
                x0 = -xb + (i + 1.5) * delta_x
                x1 = -xb + (i + 1) * delta_x
                A[i, i + 1] = (
                    -np.exp(
                        beta
                        * 0.5
                        * (
                            self.scalar_potential(x0, idx, cpu=True)
                            + self.scalar_potential(x, idx, cpu=True)
                            - 2 * self.scalar_potential(x1, idx, cpu=True)
                        )
                    )
                    / delta_x**2
                )
                A[i, i] = (
                    A[i, i]
                    + np.exp(
                        beta
                        * (
                            self.scalar_potential(x, idx, cpu=True)
                            - self.scalar_potential(x1, idx, cpu=True)
                        )
                    )
                    / delta_x**2
                )

        A = -A / beta
        N = int(T / delta_t)

        # Diagonal transformation
        sc_potential = self.scalar_potential(xvec, idx, cpu=True)
        D = np.diag(np.exp(beta * sc_potential / 2))
        D_inv = np.diag(np.exp(-beta * sc_potential / 2))

        psi = np.zeros([N + 1, nx])
        psi[N, :] = np.exp(-self.scalar_g(xvec, idx, cpu=True))

        # Compute running cost vector.
        # (If scalar_f were time-dependent, you might recompute this inside the loop.)
        fvec = np.array(self.scalar_f(0, xvec, idx, cpu=True))

        # Backward Euler time stepping.
        for n in range(N - 1, -1, -1):
            # The central diagonal of the matrix being inverted is modified to
            # include the running cost term: subtract beta*f(x).
            # Note: N/T equals 1/delta_t.
            band = -delta_t * np.vstack(
                [
                    np.append([0], np.diagonal(A, offset=1)),
                    np.diagonal(A, offset=0) - N / T - fvec / beta,
                    np.append(np.diagonal(A, offset=1), [0]),
                ]
            )
            psi[n, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(psi[n + 1, :])))
        
        # Recover the optimal control from the spatial gradient of log(psi).
        ut_discrete = np.zeros([N + 1, nx - 1])
        for n in range(N + 1):
            for i in range(nx - 1):
                ut_discrete[n, i] = (
                    -2
                    / beta
                    * self.sigma[idx, idx]
                    * (-np.log(psi[n, i + 1]) + np.log(psi[n, i]))
                    / delta_x
                )

        print("ut_discrete computed")
        return ut_discrete