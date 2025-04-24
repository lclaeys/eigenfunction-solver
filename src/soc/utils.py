import torch
import numpy as np

from src.energy.quadratic import QuadraticEnergy
from socmatching.SOC_matching.utils import stochastic_trajectories

def langevin_samples(
        sde,
        x,
        t,
        lmbd,
):
    """
    Generate samples using LD.
    Returns samples of same shape as x
    """
    for t0, t1 in zip(t[:-1], t[1:]):
        dt = t1-t0
        noise = torch.randn_like(x).to(x.device)
        update = sde.b(t0, x) * dt + torch.sqrt(lmbd * dt) * torch.einsum("ij,bj->bi", sde.sigma, noise)
        x = x + update

    return x

def mala_samples(sde, x, t, lmbd):
    """
    Generate samples using MALA (Metropolis Adjusted Langevin Algorithm).
    Returns samples of the same shape as x.
    
    Assumes:
      - sde.b(t, x) computes the drift at time t and position x.
      - sde.sigma is a constant diffusion matrix.
    
    The proposal is:
       x_proposal = x + drift(x)*dt + sqrt(lmbd*dt) * (sde.sigma @ noise)
    and the acceptance probability is computed using the ratio:
       alpha = min{1, exp( [energy(x) - energy(x_proposal)]
                        + [log q(x|x_proposal) - log q(x_proposal|x)] ) }.
    """
    # Precompute the inverse of sigma (assumed constant) for the proposal density.
    inv_sigma = torch.inverse(sde.sigma)
    
    if sde.confining:
        energy_sign = 1.0
    else:
        energy_sign = -1.0
    last_energy = sde.energy(x) * 2 / lmbd * energy_sign

    for t0, t1 in zip(t[:-1], t[1:]):
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

def log_normalization_constant(
    sde, x0, ts, cfg, n_batches_normalization=512, ground_truth_control=None
):
    log_weights_list = []
    weights_list = []

    if ground_truth_control is not None:
        norm_sqd_diff_mean = 0
    for k in range(n_batches_normalization):
        (
            states,
            _,
            _,
            _,
            log_path_weight_deterministic,
            log_path_weight_stochastic,
            log_terminal_weight,
            controls,
        ) = stochastic_trajectories(
            sde,
            x0,
            ts.to(x0),
            cfg.method.lmbd,
        )
        log_weights = (
            log_path_weight_deterministic
            + log_path_weight_stochastic
            + log_terminal_weight
        )
        log_weights_list.append(log_weights)

        if k % 32 == 31:
            print(f"Batch {k+1}/{n_batches_normalization} done")
    
    log_weights = torch.stack(log_weights_list, dim=1)

    print(
        f"Average and std. dev. of log_weights for all batches: {torch.mean(log_weights)} {torch.std(log_weights)}"
    )

    log_normalization_const = torch.logsumexp(torch.logsumexp(log_weights,dim=0),dim=0) - torch.log(torch.tensor([log_weights.shape[0]*log_weights.shape[1]],device=log_weights.device))

    return log_normalization_const

def compute_EMA_log(log_value, log_EMA_value, EMA_coeff=0.01, itr=0):
    itr_avg = int(np.floor(1 / EMA_coeff))
    if itr == 0:
        return log_value
    elif itr <= itr_avg:
        return torch.logsumexp(torch.stack((log_value, log_EMA_value + np.log(itr))),dim=0) - np.log(itr + 1)
    else:
        return  torch.logsumexp(torch.stack((log_value + np.log(EMA_coeff), np.log(1 - EMA_coeff) + log_EMA_value)),dim=0)

def compute_D(A,P,lmbd):
    A = -A * 2 / lmbd
    P = P * 2 / lmbd**2
    X = 0.25 * A.T @ A + P
    Lambda, U = torch.linalg.eigh(X)
    
    return -1/2 * A + U @ torch.diag(Lambda.sqrt()) @ U.T

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
    lmbda = cfg.lmbd

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
    

def stochastic_trajectories_final(
    sde,
    x0,
    t,
    lmbd,
    detach=True,
    verbose=False,
):
    with torch.no_grad():
        log_path_weight_deterministic = torch.zeros(x0.shape[0]).to(x0.device)
        log_path_weight_stochastic = torch.zeros(x0.shape[0]).to(x0.device)
        log_terminal_weight = torch.zeros(x0.shape[0]).to(x0.device)
        for t0, t1 in zip(t[:-1], t[1:]):
            dt = t1 - t0
            noise = torch.randn_like(x0).to(x0.device)
            u0 = sde.control(t0, x0, verbose=verbose)

            update = (
                sde.b(t0, x0) + torch.einsum("ij,bj->bi", sde.sigma, u0)
            ) * dt + torch.sqrt(lmbd * dt) * torch.einsum("ij,bj->bi", sde.sigma, noise)
            x0 = x0 + update

            log_path_weight_deterministic = (
                log_path_weight_deterministic
                + dt / lmbd * (-sde.f(t0, x0) - 0.5 * torch.sum(u0**2, dim=1))
            )
            log_path_weight_stochastic = log_path_weight_stochastic + torch.sqrt(
                dt / lmbd
            ) * (-torch.sum(u0 * noise, dim=1))

        log_terminal_weight = -sde.g(x0) / lmbd

        return (
                x0,
                log_path_weight_deterministic,
                log_path_weight_stochastic,
                log_terminal_weight,
            )
    

def control_objective(
    sde, x0, ts, lmbd, batch_size, total_n_samples=65536, verbose=False
):
    n_batches = int(total_n_samples // batch_size)
    effective_n_samples = n_batches * batch_size
    for k in range(n_batches):
        state0 = x0.repeat(batch_size, 1)
        (
            _,
            log_path_weight_deterministic,
            _,
            log_terminal_weight,
        ) = stochastic_trajectories_final(
            sde,
            state0,
            ts.to(state0),
            lmbd,
            verbose=verbose,
        )
        if k == 0:
            ctrl_losses = -lmbd * (log_path_weight_deterministic + log_terminal_weight)
        else:
            ctrl_loss = -lmbd * (log_path_weight_deterministic + log_terminal_weight)
            ctrl_losses = torch.cat((ctrl_losses, ctrl_loss), 0)
        if k % 32 == 31:
            print(f"Batch {k+1}/{n_batches} done")
    
    mean_loss = torch.nanmean(ctrl_losses)
    std_loss = torch.nanmean((ctrl_losses - mean_loss)**2).sqrt()
    return mean_loss, std_loss