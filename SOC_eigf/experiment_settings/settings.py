"""

Define the settings of the experiments, and initialize the NeuralSDE modules. Adapted from https://github.com/facebookresearch/SOC-matching.

"""
import torch
import numpy as np

from SOC_eigf.utils import optimal_control_LQ
from SOC_eigf.models import LinearControl
from SOC_eigf.experiment_settings.OU_quadratic import OU_Quadratic
from SOC_eigf.experiment_settings.ring import Ring

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
            A = -0.2 * torch.eye(cfg.d).to(cfg.device)
            P = 0.2 * torch.eye(cfg.d).to(cfg.device)
            Q = 0.1 * torch.eye(cfg.d).to(cfg.device)
        elif cfg.setting == "OU_quadratic_stable":
            A = -1.0 * torch.eye(cfg.d).to(cfg.device)
            P = 1.0 * torch.eye(cfg.d).to(cfg.device)
            Q = 0.5 * torch.eye(cfg.d).to(cfg.device)
        elif cfg.setting == "OU_quadratic_anisotropic":
            A = -1.0 * torch.eye(cfg.d).to(cfg.device)
            P = torch.linspace(0.25,5,cfg.d).diag().to(cfg.device)
            Q = 0.5 * torch.eye(cfg.d).to(cfg.device)
         
        optimal_sde = ground_truth_control(cfg, ts, x0, sigma=sigma, A=A, P=P, Q=Q)
        neural_sde = define_neural_sde(
            cfg, ts, x0, sigma=sigma, A=A, P=P, Q=Q
        )
        return x0, sigma, optimal_sde, neural_sde

    elif cfg.setting == "ring":
        sigma = torch.eye(cfg.d).to(cfg.device)

        P = torch.zeros(cfg.d).to(cfg.device)
        P[0] = 1.0
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
