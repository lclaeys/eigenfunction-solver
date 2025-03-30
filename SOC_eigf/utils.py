"""
This code contains the following utility functions:
    - a Langevin/MALA sampler

Furthermore, it contains the following functions which are adapted from https://github.com/facebookresearch/SOC-matching:
    - Compute exact solution for LQR
    - Compute normalization constant for IDO loss (changed to log space)
    - Compute EMA (changed to log space)
    - Compute control objective
    - Compute trajectories (added function that only outputs the final states)
    - Solution for LQR
"""

import torch
import numpy as np
import heapq
from scipy.special import eval_hermitenorm
from scipy.special import factorial
from tqdm import tqdm

def mala_samples(sde, x, t, lmbd, verbose=False):
    """
    Generate samples using MALA (Metropolis Adjusted Langevin Algorithm).
    Returns samples of the same shape as x.
    
    Args:
        sde (NeuralSDE): object with energy, drift (b) and sigma attributes
        x (tensor)[N,d]: initial points
        t (tensor)[T]: times to evaluate
        lmbd (float): noise level for proposal
        verbose (bool)
    """
    # Precompute the inverse of sigma (assumed constant) for the proposal density.
    inv_sigma = torch.inverse(sde.sigma)
    
    if sde.confining:
        energy_sign = 1.0
    else:
        energy_sign = -1.0
    last_energy = sde.energy(x) * 2 / lmbd * energy_sign

    time_iterator = zip(t[:-1], t[1:])
    
    if verbose:
        time_iterator = tqdm(time_iterator, total=len(t)-1, desc="MALA Sampling")

    for t0, t1 in time_iterator:
        dt = t1 - t0
        
        # Propose a candidate move using the ULA (Euler–Maruyama) step.
        noise = torch.randn_like(x, device=x.device)
        drift_x = sde.b(t0, x) * energy_sign * 2 / lmbd
        proposal = x + drift_x * dt + torch.sqrt(lmbd * dt) * torch.einsum("ij,bj->bi", sde.sigma, noise)
        
        # Compute the drift at the proposal point (using time t1).
        drift_proposal = sde.b(t1, proposal) * energy_sign * 2 / lmbd
        
        # Compute the forward move residual:
        #   diff_forward = proposal - (x + drift_x*dt)
        diff_forward = proposal - x - drift_x * dt
        # And the reverse move residual:
        #   diff_reverse = x - (proposal + drift_proposal*dt)
        diff_reverse = x - proposal - drift_proposal * dt
        
        # To compute the (log) proposal densities (up to constants), we “whiten” these differences.
        # Here the proposal distribution is Gaussian with covariance (lmbd*dt)*(sde.sigma @ sde.sigma^T).
        # Compute the transformed differences:
        transformed_forward = torch.einsum("ij,bj->bi", inv_sigma, diff_forward)
        transformed_reverse = torch.einsum("ij,bj->bi", inv_sigma, diff_reverse)
        
        # Compute squared Mahalanobis norms.
        sq_norm_forward = (transformed_forward ** 2).sum(dim=1)
        sq_norm_reverse = (transformed_reverse ** 2).sum(dim=1)
        
        # Log-density (ignoring normalization constants that cancel out)
        log_q_forward = -0.5 * sq_norm_forward / (lmbd * dt)
        log_q_reverse = -0.5 * sq_norm_reverse / (lmbd * dt)
        
        # Compute the Metropolis log acceptance ratio.
        # Since energy(x) is the potential (i.e. -log p(x) up to constant), the difference 
        # energy(x) - energy(proposal) accounts for the target density ratio.

        proposal_energy = sde.energy(proposal) * 2 / lmbd * energy_sign

        log_alpha = (-proposal_energy + last_energy) + (log_q_reverse - log_q_forward)
        
        # Clamp log_alpha at 0 before exponentiating so that acceptance probability is ≤ 1.
        alpha = torch.exp(torch.minimum(log_alpha, torch.zeros_like(log_alpha)))
        
        # Decide whether to accept the proposal.
        # Generate a random number for each sample (assume x is [batch, dims]).
        accept = (torch.rand(x.shape[0], device=x.device) < alpha)
        # If accepted, update x to the proposal; otherwise, keep the current x.
        x = torch.where(accept[:,None], proposal, x)
        last_energy = torch.where(accept, proposal_energy, last_energy)
        
    return x

def solution_Ricatti(R_inverse, A, P, Q, t):
    FT = Q
    Ft = [FT]
    for t0, t1 in zip(t[:-1], t[1:]):
        dt = t1 - t0
        FT = FT - dt * (
            -torch.matmul(torch.transpose(A, 0, 1), FT)
            - torch.matmul(FT, A)
            + 2 * torch.matmul(torch.matmul(FT, R_inverse), FT)
            - P
        )
        Ft.append(FT)
    Ft.reverse()
    return torch.stack(Ft)


def optimal_control_LQ(sigma, A, P, Q, t):
    R_inverse = torch.matmul(sigma, torch.transpose(sigma, 0, 1))
    Ft = solution_Ricatti(R_inverse, A, P, Q, t)
    ut = -2 * torch.einsum("ij,bjk->bik", torch.transpose(sigma, 0, 1), Ft)
    return ut


def exponential_t_A(t, A):
    return torch.matrix_exp(t.unsqueeze(1).unsqueeze(2) * A.unsqueeze(0))

def compute_EMA(value, EMA_value, EMA_coeff=0.01, itr=0):
    itr_avg = int(np.floor(1 / EMA_coeff))
    if itr == 0:
        return value
    elif itr <= itr_avg:
        return (value + itr * EMA_value) / (itr + 1)
    else:
        return EMA_coeff * value + (1 - EMA_coeff) * EMA_value
