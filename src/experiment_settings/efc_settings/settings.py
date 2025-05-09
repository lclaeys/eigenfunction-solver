# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import torch
from scipy.stats import ortho_group
import numpy as np

from socmatching.SOC_matching.utils import (
    optimal_control_LQ,
    exponential_t_A,
    restricted_SOC,
)
from socmatching.SOC_matching.models import (
    LinearControl,
    ConstantControlLinear,
    LowDimControl,
    RestrictedControl,
)

from src.experiment_settings.efc_settings.OU_quadratic import OU_Quadratic
from src.experiment_settings.efc_settings.OU_linear import OU_Linear
from src.experiment_settings.efc_settings.double_well import DoubleWell
from src.experiment_settings.efc_settings.ring import Ring


def ground_truth_control(cfg, ts, x0, **kwargs):
    if (
        cfg.method.setting == "OU_quadratic_easy"
        or cfg.method.setting == "OU_quadratic_hard"
        or cfg.method.setting == "OU_quadratic_stable"
    ):
        R_inverse = torch.matmul(
            kwargs["sigma"], torch.transpose(kwargs["sigma"], 0, 1)
        )
        R = torch.inverse(R_inverse)

        ut = optimal_control_LQ(
            kwargs["sigma"], kwargs["A"], kwargs["P"], kwargs["Q"], ts
        )
        ut = LinearControl(ut, cfg.method.T)

        optimal_sde = OU_Quadratic(
            u=ut,
            lmbd=cfg.method.lmbd,
            A=kwargs["A"],
            P=kwargs["P"],
            Q=kwargs["Q"],
            sigma=kwargs["sigma"],
            T=cfg.method.T,
        )

        return optimal_sde

    elif cfg.method.setting == "OU_linear":
        exp_matrix = exponential_t_A(
            cfg.method.T - ts, torch.transpose(kwargs["A"], 0, 1)
        )
        ut = -torch.einsum(
            "aij,j->ai",
            torch.einsum(
                "ij,ajk->aik", torch.transpose(kwargs["sigma"], 0, 1), exp_matrix
            ),
            kwargs["omega"],
        )
        ut = ConstantControlLinear(ut, cfg.method.T)

        optimal_sde = OU_Linear(
            u=ut,
            lmbd=cfg.method.lmbd,
            A=kwargs["A"],
            omega=kwargs["omega"],
            sigma=kwargs["sigma"],
            T=cfg.method.T,
        )

        return optimal_sde

    elif cfg.method.setting == "double_well":
        optimal_sde = DoubleWell(
            lmbd=cfg.method.lmbd,
            kappa=kwargs["kappa"],
            nu=kwargs["nu"],
            sigma=kwargs["sigma"],
            T=cfg.method.T,
        )
        xb = 2.75
        delta_t = cfg.method.delta_t_optimal
        delta_x = cfg.method.delta_x_optimal
        ut_list = []
        for j in range(cfg.method.d):
            ut_discrete = optimal_sde.compute_reference_solution(
                T=cfg.method.T,
                delta_t=delta_t,
                xb=xb,
                delta_x=delta_x,
                lmbd=cfg.method.lmbd,
                idx=j,
            )
            print(f"ut_discrete.shape: {ut_discrete.shape}")
            ut_list.append(torch.from_numpy(ut_discrete).to(cfg.method.device))
        ut_discrete = torch.stack(ut_list, dim=2)
        print(f"ut_discrete.shape: {ut_discrete.shape}")
        print(f"torch.mean(ut_discrete): {torch.mean(ut_discrete)}")

        ut = LowDimControl(
            ut_discrete, cfg.method.T, xb, cfg.method.d, delta_t, delta_x
        )
        optimal_sde.u = ut

        return optimal_sde

def define_neural_sde(cfg, ts, x0, **kwargs):
    if (
        cfg.method.setting == "OU_quadratic_easy"
        or cfg.method.setting == "OU_quadratic_hard"
        or cfg.method.setting == "OU_quadratic_stable"
    ):  
        neural_sde = OU_Quadratic(
            device=cfg.method.device,
            dim=cfg.method.d,
            hdims=cfg.arch.hdims,
            prior=cfg.arch.prior,
            joint=cfg.arch.joint,
            lmbd=cfg.method.lmbd,
            A=kwargs["A"],
            P=kwargs["P"],
            Q=kwargs["Q"],
            sigma=kwargs["sigma"],
            T=cfg.method.T,
            k=cfg.method.k,
        )
    elif cfg.method.setting == "OU_linear":
        neural_sde = OU_Linear(
            device=cfg.method.device,
            dim=cfg.method.d,
            hdims=cfg.arch.hdims,
            hdims_M=cfg.arch.hdims_M,
            lmbd=cfg.method.lmbd,
            A=kwargs["A"],
            omega=kwargs["omega"],
            sigma=kwargs["sigma"],
            gamma=cfg.method.gamma,
            scaling_factor_nabla_V=cfg.method.scaling_factor_nabla_V,
            scaling_factor_M=cfg.method.scaling_factor_M,
            T=cfg.method.T,
        )
    elif cfg.method.setting == "double_well":
        neural_sde = DoubleWell(
            device=cfg.method.device,
            dim=cfg.method.d,
            hdims=cfg.arch.hdims,
            prior=cfg.arch.prior,
            joint=cfg.arch.joint,
            lmbd=cfg.method.lmbd,
            kappa=kwargs["kappa"],
            nu=kwargs["nu"],
            sigma=kwargs["sigma"],
            T = cfg.method.T,
            k = cfg.method.k,
        )
    elif cfg.method.setting == "ring":
        neural_sde = Ring(
            device=cfg.method.device,
            dim=cfg.method.d,
            hdims=cfg.arch.hdims,
            prior=cfg.arch.prior,
            joint=cfg.arch.joint,
            lmbd=cfg.method.lmbd,
            radius=kwargs['radius'],
            scale=kwargs['scale'],
            P=kwargs["P"],
            Q=kwargs["Q"],
            sigma=kwargs["sigma"],
            T = cfg.method.T,
            k = cfg.method.k,
        )
    neural_sde.initialize_models()
    return neural_sde

def rand_spd(cfg, seed):
    np.random.seed(seed)
    rand_U = torch.tensor(ortho_group.rvs(dim = cfg.method.d),dtype=torch.float32).to(cfg.method.device)
    rand_L = torch.diag(((torch.rand(cfg.method.d) + 1) / 2)).to(cfg.method.device)
    rand_A = rand_U @ rand_L @ rand_U.T
    return (rand_A + rand_A.T) / 2

def define_variables(cfg, ts, compute_optimal=True):
    if (
        cfg.method.setting == "OU_quadratic_easy"
        or cfg.method.setting == "OU_quadratic_hard"
        or cfg.method.setting == "OU_quadratic_stable"
    ):
        if cfg.method.d == 2:
            x0 = torch.tensor([0.4, 0.6]).to(cfg.method.device)
        else:
            x0 = 0.5 * torch.randn(cfg.method.d).to(cfg.method.device)
        print(f"x0: {x0}")
        sigma = torch.eye(cfg.method.d).to(cfg.method.device)
        if cfg.method.setting == "OU_quadratic_hard":
            A = 1.0 * torch.eye(cfg.method.d).to(cfg.method.device)
            P = 1.0 * torch.eye(cfg.method.d).to(cfg.method.device)
            Q = 0.5 * torch.eye(cfg.method.d).to(cfg.method.device)
        elif cfg.method.setting == "OU_quadratic_easy":
            A = -0.2 * torch.eye(cfg.method.d).to(cfg.method.device)
            P = 0.2 * torch.eye(cfg.method.d).to(cfg.method.device)
            Q = 0.1 * torch.eye(cfg.method.d).to(cfg.method.device)
        elif cfg.method.setting == "OU_quadratic_stable":
            A = -1.0 * torch.eye(cfg.method.d).to(cfg.method.device)
            P = 1.0 * torch.eye(cfg.method.d).to(cfg.method.device)
            Q = 0.5 * torch.eye(cfg.method.d).to(cfg.method.device)
         
        optimal_sde = ground_truth_control(cfg, ts, x0, sigma=sigma, A=A, P=P, Q=Q)
        neural_sde = define_neural_sde(
            cfg, ts, x0, sigma=sigma, A=A, P=P, Q=Q
        )
        return x0, sigma, optimal_sde, neural_sde

    elif cfg.method.setting == "OU_linear":
        x0 = torch.zeros(cfg.method.d).to(cfg.method.device)
        nu = 0.1
        xi = nu * torch.randn(cfg.method.d, cfg.method.d).to(cfg.method.device)
        omega = torch.ones(cfg.method.d).to(cfg.method.device)
        A = -torch.eye(cfg.method.d).to(cfg.method.device) + xi
        sigma = torch.eye(cfg.method.d).to(cfg.method.device) + xi

        optimal_sde = ground_truth_control(cfg, ts, x0, sigma=sigma, omega=omega, A=A)
        neural_sde = define_neural_sde(
            cfg, ts, x0, sigma=sigma, omega=omega, A=A
        )
        return x0, sigma, optimal_sde, neural_sde

    elif cfg.method.setting == "double_well":
        print(f"double_well")
        x0 = torch.zeros(cfg.method.d).to(cfg.method.device)

        kappa_i = 5
        nu_i = 3
        kappa = torch.ones(cfg.method.d).to(cfg.method.device)
        nu = torch.ones(cfg.method.d).to(cfg.method.device)
        kappa[0] = kappa_i
        kappa[1] = kappa_i
        kappa[2] = kappa_i
        nu[0] = nu_i
        nu[1] = nu_i
        nu[2] = nu_i

        sigma = torch.eye(cfg.method.d).to(cfg.method.device)

        optimal_sde = None
        if compute_optimal:
            optimal_sde = ground_truth_control(cfg, ts, x0, sigma=sigma, kappa=kappa, nu=nu)
        
        neural_sde = define_neural_sde(
            cfg, ts, x0, sigma=sigma, kappa=kappa, nu=nu
        )

        return x0, sigma, optimal_sde, neural_sde

    elif cfg.method.setting=='ring':
        
        sigma = torch.eye(cfg.method.d).to(cfg.method.device)

        P = 1.0 * torch.ones(cfg.method.d).to(cfg.method.device)
        Q = 0.5 * torch.eye(cfg.method.d).to(cfg.method.device)
        scale = 1.0
        radius = 5.0

        x0 = torch.ones(cfg.method.d).to(cfg.method.device) * radius / np.sqrt(2 * cfg.method.d)

        optimal_sde = None

        neural_sde = define_neural_sde(
            cfg, ts, x0, sigma=sigma, P=P, Q=Q, scale=scale, radius=radius
        )

        return x0, sigma, optimal_sde, neural_sde
    
    elif cfg.method.setting == "ring_particles":

        sigma = torch.eye(cfg.method.d).to(cfg.method.device)

        P = 1.0 * torch.ones(cfg.method.d).to(cfg.method.device)
        R = 1.0
        scale = 1.0
        radius = 5.0

        x0 = torch.ones(cfg.method.d).to(cfg.method.device) * radius / np.sqrt(2 * cfg.method.d)

        optimal_sde = None

        neural_sde = define_neural_sde(
            cfg, ts, x0, sigma=sigma, P=P, Q=Q, scale=scale, radius=radius
        )

        return x0, sigma, optimal_sde, neural_sde
