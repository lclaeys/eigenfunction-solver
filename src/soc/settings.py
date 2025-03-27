import torch

from src.energy.quadratic import QuadraticEnergy

def get_energy(cfg, neural_sde):
    """
    Return energy model for a given cfg and SDE
    """
    if cfg.method.setting[:12] == "OU_quadratic":
        
        A_eq = -2 * neural_sde.A / cfg.method.lmbd
        energy = QuadraticEnergy(A_eq)

        return energy
    else:
        raise NotImplementedError
    
def get_Rfunc(cfg, neural_sde):
    """
    Returns the function R for a given cfg and SDE.

    Function R should take batched inputs.
    """
    if cfg.method.setting[:12] == "OU_quadratic":
        P = neural_sde.P
        return lambda x: 2/cfg.method.lmbd**2 * torch.einsum('ij, ij -> i',x @ P, x)

    else:
        raise NotImplementedError