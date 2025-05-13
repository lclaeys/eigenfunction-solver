"""

Model set up for double well setting.

"""
import torch
import numpy as np
from scipy.linalg import solve_banded


from SOC_eigf_old2.method import NeuralSDE

class DoubleWell(NeuralSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        u=None,
        lmbd=1.0,
        sigma=torch.eye(2),
        kappa = 1.0,
        nu = 1.0,
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
            return torch.zeros(x.shape[0]).to(x.device)
        elif len(x.shape) == 3:
            return torch.zeros(x.shape[0], x.shape[1]).to(x.device)

    # Gradient of final cost
    def nabla_g(self, x):
        return torch.zeros_like(x)
        
    # Scalar potential
    def scalar_potential(self, x, idx, cpu=False):
        if cpu:
            return self.kappa.cpu().numpy()[idx] * (x**2 - 1) ** 2
        else:
            return self.kappa[idx] * (x**2 - 1) ** 2
        
    def scalar_potential_grad(self, x, idx, cpu=False):
        if cpu:
            return 4*self.kappa.cpu().numpy()[idx] * (x**3 - x) 
        else:
            return 4*self.kappa[idx] * (x**3 - x)

    # scalar potential of transformed Schrodinger operator
    def scalar_V(self, x, idx, cpu=False):
        if cpu:
            return self.kappa.cpu().numpy()[idx] * ( -1/self.lmbd * (4*(3*x**2-1)) + 1/self.lmbd**2 * (4*(x**3-x))**2 + 2/self.lmbd**2 * self.scalar_f(x, idx, cpu))
        else:
            return self.kappa[idx] * ( -1/self.lmbd * (4*(2*x**2-1)) + 1/self.lmbd**2 * (4*(x**3-x))**2 + 2/self.lmbd**2 * self.scalar_f(x, idx,cpu))

    # Running cost
    def scalar_f(self, x, idx,cpu=False):
        if cpu:
            return self.nu.cpu().numpy()[idx] * (x**2 - 1) ** 2
        else:
            return self.nu[idx] * (x**2 - 1) ** 2

    # Final cost
    def scalar_g(self, x, idx, cpu=False):
        if cpu:
            return np.zeros_like(x)
        else:
            return torch.zeros_like(x).to(x.device)
        
    def compute_reference_solution(
        self, T=1.0, delta_t=0.005, delta_x=0.005, xb=2.5, lmbd=1.0, idx=0
    ):
        nx = int(2.0 * xb / delta_x) - 1

        xvec_mid = np.linspace(-xb, xb, nx+2, endpoint=True)[:-1] + delta_x / 2
        xvec = np.linspace(-xb,xb,nx+2,endpoint=True)[1:-1]

        energy_vec_mid = self.scalar_potential(xvec_mid, idx, cpu=True)
        energy_vec = self.scalar_potential(xvec, idx, cpu=True)

        #M = np.array([np.concatenate([[0],weight_vec_mid[1:-1]]), weight_vec_mid[1:] + weight_vec_mid[:-1], np.concatenate([weight_vec_mid[1:-1],[0]])]) * delta_x / 4
        #A_base = np.array([np.concatenate([[0],-weight_vec_mid[1:-1]]), weight_vec_mid[1:] + weight_vec_mid[:-1], np.concatenate([-weight_vec_mid[1:-1],[0]])]) / delta_x
        
        D = np.exp(energy_vec / self.lmbd)
        D_inv = np.exp(-energy_vec / self.lmbd)

        N = int(T / delta_t)

        v = np.zeros([N + 1, nx])
        v[0, :] = np.exp(-(self.scalar_g(xvec, idx, cpu=True)) / self.lmbd)

        A = 1/delta_x**2 * np.vstack(
            [np.append([0],-np.exp((energy_vec[:-1]+energy_vec[1:] - 2*energy_vec_mid[1:-1])/self.lmbd)),
             np.exp(2/self.lmbd * (energy_vec - energy_vec_mid[1:])) + np.exp(2/self.lmbd * (energy_vec - energy_vec_mid[:-1])),
             np.append(-np.exp((energy_vec[:-1]+energy_vec[1:] - 2*energy_vec_mid[1:-1])/self.lmbd),[0])]
        )

        f = self.scalar_f(xvec, idx, cpu=True)

        banded_identity = np.zeros([3,nx])
        banded_identity[1,:] = 1.0

        banded_fmult = banded_identity * f[None,:]

        left_matrix = banded_identity + self.lmbd / 2 * delta_t * (A + banded_fmult * 2/self.lmbd**2)
    
        for n in range(N):
            v[n+1,:] = D * solve_banded((1,1), 
                                    left_matrix, 
                                    D_inv * v[n])
    
        v_pairs = np.zeros([N+1,nx+1])
        v_pairs[:,1:] += v
        v_pairs[:,:-1] += v
        v_pairs_log = np.log(v_pairs)

        ut_discrete = self.lmbd * (v_pairs_log[:,1:] - v_pairs_log[:,:-1]) / delta_x

        print(f"ut_discrete computed")
        return np.flip(ut_discrete, axis=0).copy()
    
    @staticmethod
    def banded_matvec_vectorized(ab, x, l=1, u=1):
        n = x.shape[0]
        y = np.zeros_like(x)

        for diag_offset in range(-l, u + 1):
            row = u - diag_offset  # row in ab that stores this diagonal

            if diag_offset < 0:
                i = np.arange(-diag_offset, n)
                j = np.arange(0, n + diag_offset)
            else:
                i = np.arange(0, n - diag_offset)
                j = np.arange(diag_offset, n)

            y[i] += ab[row, j] * x[j]
        
        return y