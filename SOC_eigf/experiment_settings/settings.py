"""

Define the settings of the experiments, and initialize the NeuralSDE modules. Adapted from https://github.com/facebookresearch/SOC-matching.

"""
import torch
import numpy as np
from scipy.stats import ortho_group

from SOC_eigf_old2.utils import optimal_control_LQ

from SOC_eigf_old2.models import LinearControl, LowDimControl

from SOC_eigf_old2.experiment_settings.OU_quadratic import OU_Quadratic
from SOC_eigf_old2.experiment_settings.ring import Ring
from SOC_eigf_old2.experiment_settings.ring_particles import RingParticles
from SOC_eigf_old2.experiment_settings.double_well import DoubleWell


def ground_truth_control(cfg, ts, x0, **kwargs):
    if (
        cfg.setting == "OU_quadratic_easy"
        or cfg.setting == "OU_quadratic_hard"
        or cfg.setting == "OU_quadratic_stable"
        or cfg.setting == "OU_quadratic_anisotropic"
    ):
        R_inverse = torch.matmul(
            kwargs["sigma"], torch.transpose(kwargs["sigma"], 0, 1)
        )
        R = torch.inverse(R_inverse)

        ut = optimal_control_LQ(
            kwargs["sigma"], kwargs["A"], kwargs["P"], kwargs["Q"], ts
        )
        ut = LinearControl(ut, cfg.T)

        optimal_sde = OU_Quadratic(
            device=cfg.device,
            u=ut,
            dim=cfg.d,
            lmbd=cfg.lmbd,
            A=kwargs["A"],
            P=kwargs["P"],
            Q=kwargs["Q"],
            sigma=kwargs["sigma"],
            T=cfg.T,
            method=cfg.method,
            eigf_cfg=cfg.eigf,
            ido_cfg=cfg.ido
        )

        return optimal_sde
    
    elif cfg.setting == "double_well" or cfg.setting == "double_well_hard":
        if cfg.setting == "double_well":
            optimal_sde = DoubleWell(
                device=cfg.device,
                u=None,
                dim=cfg.d,
                lmbd=cfg.lmbd,
                kappa=kwargs["kappa"],
                nu=kwargs["nu"],
                sigma=kwargs["sigma"],
                T=cfg.T,
                method=cfg.method,
                eigf_cfg=cfg.eigf,
                ido_cfg=cfg.ido
            )

        xb = 2.75
        delta_t = cfg.delta_t_optimal
        delta_x = cfg.delta_x_optimal
        ut_list = []
        for j in range(cfg.d):
            ut_discrete = optimal_sde.compute_reference_solution(
                T=cfg.T,
                delta_t=delta_t,
                xb=xb,
                delta_x=delta_x,
                lmbd=cfg.lmbd,
                idx=j,
            )
            print(f"ut_discrete.shape: {ut_discrete.shape}")
            ut_list.append(torch.from_numpy(ut_discrete).to(cfg.device))
        ut_discrete = torch.stack(ut_list, dim=2)
        print(f"ut_discrete.shape: {ut_discrete.shape}")
        print(f"torch.mean(ut_discrete): {torch.mean(ut_discrete)}")

        ut = LowDimControl(
            ut_discrete, cfg.T, xb, cfg.d, delta_t, delta_x
        )
        optimal_sde.u = ut

        return optimal_sde


def define_neural_sde(cfg, ts, x0, **kwargs):
    if (
        cfg.setting == "OU_quadratic_easy"
        or cfg.setting == "OU_quadratic_hard"
        or cfg.setting == "OU_quadratic_stable"
        or cfg.setting == "OU_quadratic_anisotropic"
    ): 
        neural_sde = OU_Quadratic(
            device=cfg.device,
            u=None,
            dim=cfg.d,
            lmbd=cfg.lmbd,
            A=kwargs["A"],
            P=kwargs["P"],
            Q=kwargs["Q"],
            sigma=kwargs["sigma"],
            T=cfg.T,
            method=cfg.method,
            eigf_cfg=cfg.eigf,
            ido_cfg=cfg.ido
        )
    elif cfg.setting == 'ring':
        neural_sde = Ring(
            device=cfg.device,
            u=None,
            dim=cfg.d,
            lmbd=cfg.lmbd,
            radius=kwargs["radius"],
            scale=kwargs["scale"],
            P=kwargs["P"],
            Q=kwargs["Q"],
            sigma=kwargs["sigma"],
            T=cfg.T,
            method=cfg.method,
            eigf_cfg=cfg.eigf,
            ido_cfg=cfg.ido
        )
    elif cfg.setting == "ring_particles":
        neural_sde = RingParticles(
            device=cfg.device,
            u=None,
            dim=cfg.d,
            lmbd=cfg.lmbd,
            radius=kwargs["radius"],
            scale=kwargs["scale"],
            P=kwargs["P"],
            R=kwargs["R"],
            sigma=kwargs["sigma"],
            T=cfg.T,
            method=cfg.method,
            eigf_cfg=cfg.eigf,
            ido_cfg=cfg.ido
        )
    elif cfg.setting == "double_well":
        neural_sde = DoubleWell(
            device=cfg.device,
            u=None,
            dim=cfg.d,
            lmbd=cfg.lmbd,
            kappa=kwargs["kappa"],
            nu=kwargs["nu"],
            sigma=kwargs["sigma"],
            T=cfg.T,
            method=cfg.method,
            eigf_cfg=cfg.eigf,
            ido_cfg=cfg.ido
        )
    elif cfg.setting == "double_well_hard":
        neural_sde = DoubleWell(
            device=cfg.device,
            u=None,
            dim=cfg.d,
            lmbd=cfg.lmbd,
            kappa=kwargs["kappa"],
            nu=kwargs["nu"],
            sigma=kwargs["sigma"],
            T=cfg.T,
            method=cfg.method,
            eigf_cfg=cfg.eigf,
            ido_cfg=cfg.ido
        )
    
    neural_sde.initialize_models()
    return neural_sde

def define_variables(cfg, ts):
    if (
        cfg.setting == "OU_quadratic_easy"
        or cfg.setting == "OU_quadratic_hard"
        or cfg.setting == "OU_quadratic_stable"
        or cfg.setting == "OU_quadratic_anisotropic"
    ):
        x0 = 0.5 * torch.randn(cfg.d).to(cfg.device)
        sigma = torch.eye(cfg.d).to(cfg.device)

        if cfg.setting == "OU_quadratic_hard":
            A = 1.0 * torch.eye(cfg.d).to(cfg.device)
            P = 1.0 * torch.eye(cfg.d).to(cfg.device)
            Q = 0.5 * torch.eye(cfg.d).to(cfg.device)
        elif cfg.setting == "OU_quadratic_easy":
            rng = np.random.RandomState(0)
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)

            U = torch.tensor(ortho_group.rvs(cfg.d,random_state=rng)).float()
            a = torch.randn([cfg.d])
            p = torch.randn([cfg.d])
            A = 1.0 * torch.diag(a.exp()).to(cfg.device)
            P = (U @ torch.diag(p.exp()) @ U.T).to(cfg.device)
            Q = 0.5 * torch.eye(cfg.d).to(cfg.device)
            
        elif cfg.setting == "OU_quadratic_stable":
            A = -1.0 * torch.eye(cfg.d).to(cfg.device)
            P = 1.0 * torch.eye(cfg.d).to(cfg.device)
            Q = 0.5 * torch.eye(cfg.d).to(cfg.device)
        elif cfg.setting == "OU_quadratic_anisotropic":
            rng = np.random.RandomState(0)
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)

            U = torch.tensor(ortho_group.rvs(cfg.d,random_state=rng)).float()
            a = torch.randn([cfg.d])
            p = torch.randn([cfg.d])
            A = -1.0 * torch.diag(a.exp()).to(cfg.device)
            P = (U @ torch.diag(p.exp()) @ U.T).to(cfg.device)
            Q = 0.2 * torch.eye(cfg.d).to(cfg.device)
         
        optimal_sde = ground_truth_control(cfg, ts, x0, sigma=sigma, A=A, P=P, Q=Q)
        neural_sde = define_neural_sde(
            cfg, ts, x0, sigma=sigma, A=A, P=P, Q=Q
        )
        return x0, sigma, optimal_sde, neural_sde

    elif cfg.setting == "ring":
        sigma = torch.eye(cfg.d).to(cfg.device)

        P = torch.zeros(cfg.d).to(cfg.device)
        P[0] = 2.0
        Q = 0.0 * torch.eye(cfg.d).to(cfg.device)
        
        scale = 1.0
        radius = 5.0

        x0 = torch.zeros(cfg.d).to(cfg.device)
        x0[0] = radius / np.sqrt(2)

        optimal_sde=None

        neural_sde = define_neural_sde(
            cfg, ts, x0, sigma=sigma, P=P, Q=Q, scale=scale, radius=radius
        )

        return x0, sigma, optimal_sde, neural_sde
    
    elif cfg.setting == "ring_particles":
        sigma = torch.eye(cfg.d).to(cfg.device)

        P = torch.zeros(2).to(cfg.device)
        P[0] = 0.5
        R = 0.0
        
        scale = 1.0
        radius = 5.0

        x0 = torch.zeros(cfg.d).to(cfg.device)
        x0[::2] = radius / np.sqrt(2)

        optimal_sde=None

        neural_sde = define_neural_sde(
            cfg, ts, x0, sigma=sigma, P=P, R=R, scale=scale, radius=radius
        )

        return x0, sigma, optimal_sde, neural_sde
    
    elif cfg.setting == "double_well" or "double_well_hard":
        if cfg.setting == "double_well":
            x0 = torch.zeros(cfg.d).to(cfg.device)
        elif cfg.setting == "double_well_hard":
            x0 = -torch.ones(cfg.d).to(cfg.device)

        kappa_i = 5
        nu_i = 5
        kappa = torch.ones(cfg.d).to(cfg.device)
        nu = torch.ones(cfg.d).to(cfg.device)*3
        kappa[0] = kappa_i
        kappa[1] = kappa_i
        kappa[2] = kappa_i
        nu[0] = nu_i
        nu[1] = nu_i
        nu[2] = nu_i

        sigma = torch.eye(cfg.d).to(cfg.device)

        optimal_sde = ground_truth_control(cfg, ts, x0, sigma=sigma, kappa=kappa, nu=nu)
        
        neural_sde = define_neural_sde(
            cfg, ts, x0, sigma=sigma, kappa=kappa, nu=nu
        )

        return x0, sigma, optimal_sde, neural_sde
