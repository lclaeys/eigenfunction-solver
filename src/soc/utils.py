import torch

from src.energy.quadratic import QuadraticEnergy

def compute_D(A,P,lmbda):
    assert torch.allclose(A, torch.eye(A.shape[0],device=A.device)*A[0,0])
    Lambda, U = torch.linalg.eigh(P)

    a = torch.diag(A)
    sign = torch.where(a > 0, 1.0, -1.0)
    kappa = torch.diag(A) / lmbda * (1 + sign * torch.sqrt(1 + 2 / a**2 * Lambda))

    return U @ torch.diag(kappa) @ U.T

def exact_eigfunctions(x, m, neural_sde, cfg, return_grad = False):
    """
    Evaluate first m exact eigenfunctions of K at points x
    Args:
        x (tensor)[n,d]: evaluation points
        m (int): number of eigenfunctions to compute
        neural_sde
        cfg
        return_grad (bool): whether to return gradient
    Returns:
        fx (tensor)[n,m]: first m eigenfunction evaluations
        (Optinal) grad_fx (tensor)[n,m,d]: gradients of first m eigenfunctions
    """
    A = neural_sde.A
    P = neural_sde.P
    lmbda = cfg.method.lmbd

    D = compute_D(A, P, lmbda)
    energy = QuadraticEnergy(-  2/lmbda * A + 2*D)

    norm = (torch.linalg.det(-2/lmbda*A)** (1/2) / torch.linalg.det(-2/lmbda * A + 2 * D) ** (1/2)) ** (1/2)
    quadratic_form = torch.exp(- 1/2 * torch.einsum('ij, ij -> i',x @ D, x))
    
    if return_grad:
        wx, grad_wx = energy.exact_eigfunctions(x, m, use_scipy=False, return_grad=True)
        fx = wx * quadratic_form[:,None] / norm
        grad_fx = (grad_wx - (x @ D.T)[:,None,:] * wx[:,:,None]) * quadratic_form[:, None, None] / norm

        return fx, grad_fx
    
    else:
        wx = energy.exact_eigfunctions(x, m, use_scipy=False)
        fx = wx * quadratic_form[:,None] / norm
        
        return fx