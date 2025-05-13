"""

The main code for the solver modules. The NeuralSDE API and code for the IDO methods are adapted from https://github.com/facebookresearch/SOC-matching.

"""

import torch
import torch.nn as nn
import numpy as np
import nvidia_smi
import functorch

from SOC_eigf.models import FullyConnectedUNet, FullyConnectedUNet2, SigmoidMLP, GELUNET
from SOC_eigf.utils import mala_samples, stochastic_trajectories, compute_EMA

class NeuralSDE(nn.Module):
    """
    Class that contains the models and SDE parameters.
    """
    def __init__(
        self,
        device="cuda",
        u=None,
        dim=2,
        lmbd=1.0,
        sigma=torch.eye(2),
        T=1.0,
        method="EIGF",
        eigf_cfg=None,
        ido_cfg=None,
    ):
        super().__init__()
        assert method in ['EIGF','IDO','COMBINED','FBSDE']
        self.device = device
        self.u = u
        self.dim = dim
        self.lmbd = lmbd
        self.sigma = sigma
        self.T = T
        self.method = method

        self.k = eigf_cfg.get('k', 1)
        self.hdims_eigf = eigf_cfg.get('hdims', [128,128,128])
        self.arch_eigf = eigf_cfg.get('arch', 'SIREN')
        self.reg = eigf_cfg.get('reg', 0.0)
        
        self.hdims_ido = ido_cfg.get('hdims',[256,128,64])
        self.hdims_M = ido_cfg.get('hdims_M',[128,128])
        self.scaling_factor_nabla_V = ido_cfg.get('scaling_factor_nabla_V', 1.0)
        self.scaling_factor_M= ido_cfg.get('scaling_factor_M', 1.0)
        self.gamma = ido_cfg.get('gamma',1.0)
        self.gamma2 = ido_cfg.get('gamma2',1.0)
        self.gamma3 = ido_cfg.get('gamma3',1.0)
        self.T_cutoff = ido_cfg.get('T_cutoff',1.0)
        self.train_scalar = ido_cfg.get('train_scalar',False)
        self.use_eigval = ido_cfg.get('use_eigval', True)

        self.eigval_diff = None
        

    def control(self, t, x, verbose=False):
        if verbose:
            print(
                f"self.use_learned_control: {self.use_learned_control}, self.u: {self.u}"
            )
        if self.use_learned_control:
            if len(x.shape) == 2:
                learned_control = -torch.einsum(
                    "ij,bj->bi",
                    torch.transpose(self.sigma, 0, 1),
                    self.nabla_V(t,x),
                )
                return learned_control
            
            # x.shape = (N,B,d)
            if len(x.shape) == 3:
                learned_control = -torch.einsum(
                    "ij,abj->abi",
                    torch.transpose(self.sigma, 0, 1),
                    self.nabla_V(t,x),
                )
                return learned_control
        else:
            if self.u is None:
                return None
            else:
                return self.u(t, x)

    def initialize_models(self):
        # Use learned control in the stochastic_trajectories function
        self.use_learned_control = True

        if self.method in ['EIGF', 'COMBINED']:
            # ground state model
            self.eigf_gs_model = GELUNET(
                    dim=self.dim,
                    hdims=self.hdims_eigf,
                    k = 1,
                    reg=self.reg
                    ).to(self.device)            
            
            if self.k > 1 and self.method == "EIGF":
                self.eigf_model = GELUNET(
                    dim=self.dim,
                    hdims=self.hdims_eigf,
                    k = self.k-1,
                    reg=self.reg
                    ).to(self.device)

            # save eigvals
            self.register_buffer('eigvals', torch.zeros(self.k, device=self.device))
            self.register_buffer('inner_prods',torch.ones(self.k, device=self.device))
            self.register_buffer('norms', torch.ones(self.k, device=self.device))

            def gs_model_fn(x):
                out = self.eigf_gs_model(x)
                return out, out
            
            self.base_eigf_gs_model_jac = torch.vmap(torch.func.jacrev(gs_model_fn,has_aux=True))
            self.base_eigf_gs_model_hessian = torch.func.hessian(lambda x: gs_model_fn(x)[0])
            self.base_eigf_gs_model_laplacian = torch.vmap(lambda x: self.base_eigf_gs_model_hessian(x).diagonal(dim1=-1,dim2=-2).sum(dim=-1))

            if self.confining:
                self.eigf_gs_model_jac = self.base_eigf_gs_model_jac
                self.eigf_gs_model_laplacian = self.base_eigf_gs_model_laplacian
            else:
                def gs_model_jac(x):
                    Dfx, fx = self.base_eigf_gs_model_jac(x)
                    return Dfx - self.b(0, x)[:,None] * 2 / self.lmbd, fx
                
                def gs_model_laplacian(x):
                    return self.base_eigf_gs_model_laplacian(x) + self.Delta_E(x)[:,None] * 2 / self.lmbd
                
                self.eigf_gs_model_jac = gs_model_jac
                self.eigf_gs_model_laplacian = gs_model_laplacian

            if self.k > 1 and self.method == "EIGF":
                def eigf_model_fn(x):
                    out = self.eigf_model(x)
                    return out, out
            
                self.base_eigf_model_jac = torch.vmap(torch.func.jacrev(eigf_model_fn,has_aux=True))

                if self.confining:
                    self.eigf_model_jac = self.base_eigf_model_jac
                else:
                    def model_jac(x):
                        Dfx, fx = self.base_eigf_model_jac(x)
                        grad_Ex = -self.b(0, x)
                        return Dfx + fx[:,:,None] * grad_Ex[:,None,:] * 2 / self.lmbd, fx
                    
                    self.eigf_model_jac = model_jac
            
        if self.method == 'IDO':
            self.ido_model = FullyConnectedUNet(
                dim_in=self.dim+1,
                dim_out=self.dim,
                hdims=self.hdims_ido,
                scaling_factor=self.scaling_factor_nabla_V,
            ).to(self.device)

            self.gamma = torch.nn.Parameter(torch.tensor([self.gamma]).to(self.device))
            self.gamma2 = torch.nn.Parameter(torch.tensor([self.gamma2]).to(self.device))
            self.M = SigmoidMLP(
                dim=self.dim,
                hdims=self.hdims_M,
                gamma=self.gamma,
                scaling_factor=self.scaling_factor_M,
            ).to(self.device)

        elif self.method == 'COMBINED':
            if not self.train_scalar:
                self.ido_model = FullyConnectedUNet2(
                dim_in=self.dim+2,
                dim_out=self.dim,
                hdims=self.hdims_ido,
                scaling_factor=self.scaling_factor_nabla_V,
                ).to(self.device)
            else:
                self.ido_model = GELUNET(
                    dim=self.dim+1,
                    hdims=self.hdims_ido,
                    k = 1
                ).to(self.device)

                def ido_model_fn(x):
                    out = self.eigf_gs_model(x)
                    return out, out
                
                self.ido_model_jac = torch.vmap(torch.func.jacrev(ido_model_fn,has_aux=True))
            
            self.gamma = torch.nn.Parameter(torch.tensor([self.gamma]).to(self.device))
            self.gamma2 = torch.nn.Parameter(torch.tensor([self.gamma2]).to(self.device))
            self.M = SigmoidMLP(
                dim=self.dim,
                hdims=self.hdims_M,
                gamma=self.gamma,
                scaling_factor=self.scaling_factor_M,
            ).to(self.device)

        elif self.method=="FBSDE":
            self.fbsde_model = FullyConnectedUNet(
                dim_in=self.dim+1,
                dim_out=self.dim,
                hdims=self.hdims_ido,
                scaling_factor=self.scaling_factor_nabla_V,
            ).to(self.device)
        
    def nabla_V(self,t,x):
        """
        Output nabla_V estimate.
        """
        
        t = t.reshape(-1)
        t_rev = self.T - t
        
        if self.method == "COMBINED":
            # Determine indices that use IDO model (after T-T_cutoff)
            if len(t.shape) >= 1:
                ido_idx = torch.clamp(torch.searchsorted(t,self.T - self.T_cutoff),min=1)-1 # find cutoff index
                ido_t = t[ido_idx:]
                ido_x = x[ido_idx:]
                ido_t_rev = self.T - ido_t
            else:
                ido_idx = 0
                ido_t = t
                ido_x = x
                ido_t_rev = (self.T - ido_t).unsqueeze(0)
            
            if self.eigval_diff is None:
                self.eigval_diff = (self.eigvals[1] - self.eigvals[0]) * self.lmbd / 2

        if len(x.shape) == 2:
            if self.method == "EIGF":
                Dfx, fx = self.eigf_gs_model_jac(x)

                return -self.lmbd * Dfx[:,0,:]
            
            elif self.method == "IDO":
                x = x.reshape(-1, self.dim)
                t_expand = t.reshape(-1, 1).expand(x.shape[0], 1)
                tx = torch.cat([t_expand, x], dim=-1)
                out = self.ido_model(tx).reshape(x.shape)

                return out

            elif self.method == "FBSDE":
                x = x.reshape(-1, self.dim)
                t_expand = t.reshape(-1, 1).expand(x.shape[0], 1)
                tx = torch.cat([t_expand, x], dim=-1)
                out = self.fbsde_model(tx).reshape(x.shape)

                return out
            
            elif self.method == "COMBINED":
                Dfx, fx = self.eigf_gs_model_jac(x)
                eigf_control = -self.lmbd * Dfx[:,0,:]

                if not self.train_scalar:
                    ido_ts_expand = ido_t_rev.reshape(-1, 1).expand(ido_x.shape[0], 1)
                    ido_ttx = torch.cat([ido_ts_expand, torch.exp(-ido_ts_expand * self.eigval_diff), ido_x], dim=-1)
                    
                    ido_control = self.ido_model(ido_ttx)
                
                if not self.use_eigval:
                    self.eigval_diff = 0

                return torch.cat([eigf_control[:ido_idx], torch.exp(-ido_ts_expand * self.eigval_diff) * ido_control + eigf_control[ido_idx:]], dim=0)
        
        if len(x.shape) == 3:
            x_reshaped = x.reshape(-1,self.dim)

            if self.method == "EIGF":
                Dfx, fx = self.eigf_gs_model_jac(x_reshaped)
                Dfx = torch.reshape(Dfx, (x.shape[0],x.shape[1], 1, x.shape[2]))

                return -self.lmbd * Dfx[:,:,0,:]
            
            elif self.method == "IDO":
                ts_repeat = t.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], 1)
                tx = torch.cat([ts_repeat, x], dim=-1)
                tx_reshape = torch.reshape(tx, (-1, tx.shape[2]))

                out = self.ido_model(tx_reshape)
                reshaped_out = torch.reshape(out, x.shape)

                return reshaped_out
            
            elif self.method == "FBSDE":
                ts_repeat = t.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], 1)
                tx = torch.cat([ts_repeat, x], dim=-1)
                tx_reshape = torch.reshape(tx, (-1, tx.shape[2]))

                out = self.fbsde_model(tx_reshape)
                reshaped_out = torch.reshape(out, x.shape)

                return reshaped_out
            
            elif self.method == "COMBINED":
                Dfx, fx = self.eigf_gs_model_jac(x_reshaped)
                Dfx = torch.reshape(Dfx, (x.shape[0],x.shape[1], 1, x.shape[2]))

                eigf_control = -self.lmbd * Dfx[:,:,0,:]

                if not self.train_scalar:
                    ido_ts_repeat = ido_t_rev.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], 1)
                    ido_ttx = torch.cat([ido_ts_repeat, torch.exp(- ido_ts_repeat * self.eigval_diff), ido_x], dim=-1)
                    ido_ttx_reshape = torch.reshape(ido_ttx, (-1, ido_ttx.shape[2]))

                    fx = self.ido_model(ido_ttx_reshape)
                    ido_control = torch.reshape(fx, (ido_x.shape[0],ido_x.shape[1], ido_x.shape[2]))

                if not self.use_eigval:
                    self.eigval_diff = 0

                return torch.cat([eigf_control[:ido_idx], torch.exp(- ido_ts_repeat * self.eigval_diff) * ido_control + eigf_control[ido_idx:]], dim=0)


class SOC_Solver(nn.Module):
    """
    Class for the solver, used for computing the loss.

    solver_cfg:
        langevin_burnin_steps: number of sampling steps for burn-in phase
        langevin_sample_steps: number of sampling steps per iteration
        langevin_dt: timestep used for Langevin sampling
        beta: beta of the loss function (when relevant)
        eigf_loss: which loss to use for the eigenfunction ['pinn', 'ritz', 'var', 'rel']
        ido_algorithm: which loss to use for the IDO algorithm
        nsamples: how many samples to use in eigf method
    """

    def __init__(
        self,
        neural_sde,
        x0,
        ut,
        T=1.0,
        num_steps=100,
        lmbd=1.0,
        d=2,
        sigma=torch.eye(2),
        solver_cfg=None
    ):
        super().__init__()
        self.device = neural_sde.device
        self.dim = neural_sde.dim
        self.x0 = x0
        self.ut = ut
        self.T = T
        self.ts = torch.linspace(0,T, num_steps+1).to(self.device)
        self.dt = T / num_steps
        self.num_steps = num_steps
        self.lmbd = lmbd
        self.d = d
        self.y0 = torch.nn.Parameter(torch.randn(1, device=x0.device))
        self.sigma = sigma

        self.langevin_burnin_steps = solver_cfg.get('langevin_burnin_steps', 1000)
        self.langevin_sample_steps = solver_cfg.get('langevin_sample_steps', 100)
        self.langevin_dt = solver_cfg.get('langevin_dt', 0.01)
        self.beta = solver_cfg.get('beta', 0.1)
        self.eigf_loss = solver_cfg.get('eigf_loss', 'ritz')
        self.ido_algorithm = solver_cfg.get('ido_algorithm', 'variance')

        self.nsamples = solver_cfg.get('nsamples',65536)
        self.samples = None
        self.pinn_eval_points = None

        self.neural_sde = neural_sde
        
    def control(self, t0, x0):
        return self.neural_sde.control(t0, x0)
    
    def update_samples(self, verbose):
        if self.samples is None:
            if verbose:
                print('Burning in Langevin...')
            self.samples = self.x0.repeat(self.nsamples,1).detach_()

            burnin_langevin_ts = torch.linspace(0,self.langevin_dt * self.langevin_burnin_steps, self.langevin_burnin_steps+1).to(self.x0.device)
            self.sample_langevin_ts = torch.linspace(0,self.langevin_dt * self.langevin_sample_steps, self.langevin_sample_steps+1).to(self.x0.device)
            self.samples = mala_samples(self.neural_sde,self.samples,burnin_langevin_ts,self.lmbd,verbose=verbose)

            shift = self.samples.mean(dim=0)
            scale = self.samples.std(dim=0)
            
            if verbose:
                print(f'Completed burn-in. Samples mean {shift}, std {scale}')
            
            with torch.no_grad():
                if hasattr(self.neural_sde.eigf_gs_model,'shift'):
                    self.neural_sde.eigf_gs_model.shift.data.copy_(shift)
                    self.neural_sde.eigf_gs_model.scale.data.copy_(scale)
        
        else:
            self.samples = mala_samples(self.neural_sde,self.samples,self.sample_langevin_ts,self.lmbd).detach_()

    """
    Loss for ground state, parametrized as phi = exp(f)
    """
    def gs_loss(
        self,
        verbose=False
    ):
        if self.eigf_loss not in ['rel','pinn']:
            Dfx, fx = self.neural_sde.eigf_gs_model_jac(self.samples)
        else:
            Dfx, fx = self.neural_sde.eigf_gs_model_jac(self.samples)

        sq_f_norm = (torch.logsumexp(fx[:,None,:] + fx[:,:,None], dim = 0).exp() / fx.shape[0]).clip(min=1e-4)
        
        if self.eigf_loss == 'var':
            orth_loss = (1-sq_f_norm)**2
        else:
            orth_loss = sq_f_norm.log()**2

        if self.eigf_loss in ['var','ritz']:
            Rx = self.neural_sde.f(None,self.samples) * 2 / self.lmbd**2

            grad_norm = torch.norm(Dfx, dim=2,p=2).clip(min=1e-8)
            
            sq_grad_norm = torch.logsumexp(2*fx[:,0] + 2*grad_norm[:,0].log(),dim=0).exp() / fx.shape[0]

            Rx = torch.clip(torch.abs(Rx),min=1e-8) * torch.sign(Rx)
            R_pos_idx = Rx.squeeze() > 0
            
            R_norm = (torch.logsumexp(2*fx[R_pos_idx,0] + Rx[R_pos_idx].log(),dim=0).exp() - torch.logsumexp(2*fx[~R_pos_idx,0] + (-Rx[~R_pos_idx]).log(),dim=0).exp()) / fx.shape[0]

            var_loss = sq_grad_norm + R_norm

            if self.eigf_loss == 'var':
                self.neural_sde.eigvals[0] = 2/self.beta * (1-sq_f_norm.detach())
                return (
                    self.beta*var_loss + orth_loss, 
                    var_loss, 
                    orth_loss, 
                    (fx - 1/2*sq_f_norm.log()).detach(), 
                    Dfx.detach()
                )
            
            elif self.eigf_loss == "ritz":
                self.neural_sde.eigvals[0] = (var_loss / sq_f_norm).detach()
                return (
                    var_loss / sq_f_norm + orth_loss, 
                    var_loss / sq_f_norm, 
                    orth_loss, 
                    fx.detach(), 
                    Dfx.detach(),
                )
            
        else:
            grad_norm = torch.norm(Dfx, dim=2,p=2).clip(min=1e-8)
            Rx = self.neural_sde.f(None,self.samples) * 2 / self.lmbd**2
            
            Deltafx = self.neural_sde.eigf_gs_model_laplacian(self.samples)
            
            grad_Ex = -self.neural_sde.b(0, self.samples)
            
            Lfx = - Deltafx - grad_norm**2 + torch.einsum('ij,ikj->ik',grad_Ex,Dfx) * 2 / self.lmbd + Rx.unsqueeze(1)

            if self.eigf_loss == 'rel':
                sq_diff = ((Lfx - self.neural_sde.eigvals[0])**2).clip(min=1e-8)
                rel_loss = torch.logsumexp(sq_diff.log(),dim=0).exp() / fx.shape[0]

                return (
                    rel_loss + orth_loss,
                    rel_loss, 
                    orth_loss, 
                    fx.detach(), 
                    Dfx.detach()
                )
            
            elif self.eigf_loss == "pinn":
                sq_diff = ((Lfx - self.neural_sde.eigvals[0])**2).clip(min=1e-8)
                pinn_loss = torch.logsumexp(sq_diff.log() + 2*fx,dim=0).exp() / fx.shape[0]

                return (
                    pinn_loss / sq_f_norm + orth_loss, 
                    pinn_loss, 
                    orth_loss, 
                    fx.detach(), 
                    Dfx.detach(),
                )
            
            elif self.eigf_loss == "log_rel":
                ratio = (Lfx / self.neural_sde.eigvals[0]).clip(min=1e-6, max = 1e6)
                sq_diff = (ratio.log()**2).clip(min=1e-8)
                log_rel_loss = torch.logsumexp(sq_diff.log(),dim=0).exp() / fx.shape[0]

                return (
                    log_rel_loss + orth_loss,
                    log_rel_loss, 
                    orth_loss, 
                    fx.detach(), 
                    Dfx.detach()
                )
    
    """
    Loss for excited states
    """
    def es_loss(
        self,
        gs_fx,
        gs_Dfx,
    ):
        Dfx, fx = self.neural_sde.eigf_model_jac(self.samples)

        importance_weight = torch.ones(self.samples.shape[0], device=self.device)

        Rx = self.neural_sde.f(None,self.samples) * 2 / self.lmbd**2

        grad_norm = torch.norm(Dfx, dim=2,p=2)

        f_cov = torch.mean(fx[:,None,:] * fx[:,:,None] * importance_weight[:,None,None],dim=0)

        if (1 - self.beta * self.neural_sde.eigvals[0] / 2) <= 0:
            self.beta = 1/self.neural_sde.eigvals[0]

        gs_cov = torch.mean((gs_fx).exp() * fx * importance_weight[:,None],dim=0) * (1 - self.beta * self.neural_sde.eigvals[0] / 2).sqrt()

        orth_loss = ((f_cov - torch.eye(f_cov.shape[0],device=self.device))**2).sum() + 2*(gs_cov**2).sum()
        
        sq_grad_norm = (grad_norm**2 * importance_weight[:,None]).mean(dim=0).sum()
                
        R_norm = torch.mean(fx**2 * Rx[:,None] * importance_weight[:,None],dim=0).sum()

        var_loss = sq_grad_norm + R_norm

        self.neural_sde.eigvals[1:] = 2/self.beta * (1-f_cov.diag().detach()).clip(min=1e-5)
        return (
            self.beta*var_loss + orth_loss, 
            var_loss, 
            orth_loss, 
            fx.detach(), 
            Dfx.detach()
        )
    
    """
    Loss for IDO algorithms
    """
    def ido_loss(
        self,
        log_normalization_const=0.0,
        state0=None,
        verbose=False
    ):
        algorithm = self.ido_algorithm
        d = self.dim
        detach = algorithm != "rel_entropy"
        self.num_steps = self.ts.shape[0] - 1
        (
            states,
            noises,
            log_path_weight_deterministic,
            log_path_weight_stochastic,
            log_terminal_weight,
            controls,
        ) = stochastic_trajectories(
            self.neural_sde,
            state0,
            self.ts.to(state0),
            self.lmbd,
            detach=detach,
        )

        log_weight = log_path_weight_deterministic + log_path_weight_stochastic + log_terminal_weight - log_normalization_const
        weight = torch.exp(log_weight)

        if algorithm == "rel_entropy":
            ctrl_losses = -self.lmbd * (
                log_path_weight_deterministic + log_terminal_weight
            )
            objective = torch.mean(ctrl_losses)
            weight = weight.detach()
            learned_control = controls.detach()
        else:
            ts_repeat = self.ts.unsqueeze(1).unsqueeze(2).repeat(1, states.shape[1], 1)
            tx = torch.cat([ts_repeat, states], dim=-1)
            tx_reshape = torch.reshape(tx, (-1, tx.shape[2]))

        if algorithm == "SOCM_const_M":
            sigma_inverse_transpose = torch.transpose(torch.inverse(self.sigma), 0, 1)
            least_squares_target_integrand_term_1 = (
                self.neural_sde.nabla_f(self.ts[0], states)
            )[:-1, :, :]
            least_squares_target_integrand_term_2 = -np.sqrt(self.lmbd) * torch.einsum(
                "abij,abj->abi",
                self.neural_sde.nabla_b(self.ts[0], states)[:-1, :, :, :],
                torch.einsum("ij,abj->abi", sigma_inverse_transpose, noises),
            )
            least_squares_target_integrand_term_3 = -torch.einsum(
                "abij,abj->abi",
                self.neural_sde.nabla_b(self.ts[0], states)[:-1, :, :, :],
                torch.einsum("ij,abj->abi", sigma_inverse_transpose, controls),
            )
            least_squares_target_terminal = self.neural_sde.nabla_g(states[-1, :, :])

            dts = self.ts[1:] - self.ts[:-1]
            least_squares_target_integrand_term_1_times_dt = torch.cat(
                (
                    torch.zeros_like(
                        least_squares_target_integrand_term_1[0, :, :]
                    ).unsqueeze(0),
                    least_squares_target_integrand_term_1
                    * dts.unsqueeze(1).unsqueeze(2),
                ),
                0,
            )
            least_squares_target_integrand_term_2_times_sqrt_dt = torch.cat(
                (
                    torch.zeros_like(
                        least_squares_target_integrand_term_2[0, :, :]
                    ).unsqueeze(0),
                    least_squares_target_integrand_term_2
                    * torch.sqrt(dts).unsqueeze(1).unsqueeze(2),
                ),
                0,
            )
            least_squares_target_integrand_term_3_times_dt = torch.cat(
                (
                    torch.zeros_like(
                        least_squares_target_integrand_term_3[0, :, :]
                    ).unsqueeze(0),
                    least_squares_target_integrand_term_3
                    * dts.unsqueeze(1).unsqueeze(2),
                ),
                0,
            )

            cumulative_sum_least_squares_term_1 = torch.sum(
                least_squares_target_integrand_term_1_times_dt, dim=0
            ).unsqueeze(0) - torch.cumsum(
                least_squares_target_integrand_term_1_times_dt, dim=0
            )
            cumulative_sum_least_squares_term_2 = torch.sum(
                least_squares_target_integrand_term_2_times_sqrt_dt, dim=0
            ).unsqueeze(0) - torch.cumsum(
                least_squares_target_integrand_term_2_times_sqrt_dt, dim=0
            )
            cumulative_sum_least_squares_term_3 = torch.sum(
                least_squares_target_integrand_term_3_times_dt, dim=0
            ).unsqueeze(0) - torch.cumsum(
                least_squares_target_integrand_term_3_times_dt, dim=0
            )
            least_squares_target = (
                cumulative_sum_least_squares_term_1
                + cumulative_sum_least_squares_term_2
                + cumulative_sum_least_squares_term_3
                + least_squares_target_terminal.unsqueeze(0)
            )
            control_learned = self.control(self.ts, states)
            control_target = -torch.einsum(
                "ij,...j->...i", torch.transpose(self.sigma, 0, 1), least_squares_target
            )

            objective = torch.sum(
                (control_learned - control_target) ** 2
                * weight.unsqueeze(0).unsqueeze(2)
            ) / (states.shape[0] * states.shape[1])

        if algorithm == "SOCM_exp":
            sigma_inverse_transpose = torch.transpose(torch.inverse(self.sigma), 0, 1)
            exp_factor = torch.exp(-self.gamma * self.ts)
            identity = torch.eye(d).to(self.x0.device)
            least_squares_target_integrand_term_1 = (
                exp_factor.unsqueeze(1).unsqueeze(2)
                * self.neural_sde.nabla_f(self.ts[0], states)
            )[:-1, :, :]
            least_squares_target_integrand_term_2 = exp_factor[:-1].unsqueeze(
                1
            ).unsqueeze(2) * (
                -np.sqrt(self.lmbd)
                * torch.einsum(
                    "abij,abj->abi",
                    self.neural_sde.nabla_b(self.ts[0], states)[:-1, :, :, :]
                    + self.gamma * identity,
                    torch.einsum("ij,abj->abi", sigma_inverse_transpose, noises),
                )
            )
            least_squares_target_integrand_term_3 = exp_factor[:-1].unsqueeze(
                1
            ).unsqueeze(2) * (
                -torch.einsum(
                    "abij,abj->abi",
                    self.neural_sde.nabla_b(self.ts[0], states)[:-1, :, :, :]
                    + self.gamma * identity,
                    torch.einsum("ij,abj->abi", sigma_inverse_transpose, controls),
                )
            )
            least_squares_target_terminal = torch.exp(
                -self.gamma * (self.T - self.ts)
            ).unsqueeze(1).unsqueeze(2) * self.neural_sde.nabla_g(
                states[-1, :, :]
            ).unsqueeze(
                0
            )

            dts = self.ts[1:] - self.ts[:-1]
            least_squares_target_integrand_term_1_times_dt = torch.cat(
                (
                    torch.zeros_like(
                        least_squares_target_integrand_term_1[0, :, :]
                    ).unsqueeze(0),
                    least_squares_target_integrand_term_1
                    * dts.unsqueeze(1).unsqueeze(2),
                ),
                0,
            )
            least_squares_target_integrand_term_2_times_sqrt_dt = torch.cat(
                (
                    torch.zeros_like(
                        least_squares_target_integrand_term_2[0, :, :]
                    ).unsqueeze(0),
                    least_squares_target_integrand_term_2
                    * torch.sqrt(dts).unsqueeze(1).unsqueeze(2),
                ),
                0,
            )
            least_squares_target_integrand_term_3_times_dt = torch.cat(
                (
                    torch.zeros_like(
                        least_squares_target_integrand_term_3[0, :, :]
                    ).unsqueeze(0),
                    least_squares_target_integrand_term_3
                    * dts.unsqueeze(1).unsqueeze(2),
                ),
                0,
            )

            inv_exp_factor = 1 / exp_factor
            cumsum_least_squares_term_1 = inv_exp_factor.unsqueeze(1).unsqueeze(2) * (
                torch.sum(
                    least_squares_target_integrand_term_1_times_dt, dim=0
                ).unsqueeze(0)
                - torch.cumsum(least_squares_target_integrand_term_1_times_dt, dim=0)
            )
            cumsum_least_squares_term_2 = inv_exp_factor.unsqueeze(1).unsqueeze(2) * (
                torch.sum(
                    least_squares_target_integrand_term_2_times_sqrt_dt, dim=0
                ).unsqueeze(0)
                - torch.cumsum(
                    least_squares_target_integrand_term_2_times_sqrt_dt, dim=0
                )
            )
            cumsum_least_squares_term_3 = inv_exp_factor.unsqueeze(1).unsqueeze(2) * (
                torch.sum(
                    least_squares_target_integrand_term_3_times_dt, dim=0
                ).unsqueeze(0)
                - torch.cumsum(least_squares_target_integrand_term_3_times_dt, dim=0)
            )

            least_squares_target = (
                cumsum_least_squares_term_1
                + cumsum_least_squares_term_2
                + cumsum_least_squares_term_3
                + least_squares_target_terminal
            )
            control_learned = self.control(self.ts, states)
            control_target = -torch.einsum(
                "ij,...j->...i", torch.transpose(self.sigma, 0, 1), least_squares_target
            )

            objective = torch.sum(
                (control_learned - control_target) ** 2
                * weight.unsqueeze(0).unsqueeze(2)
            ) / (states.shape[0] * states.shape[1])

        if algorithm == "SOCM":
            sigma_inverse_transpose = torch.transpose(torch.inverse(self.sigma), 0, 1)
            identity = torch.eye(self.dim).to(self.x0.device)

            
            sum_M = lambda t, s: self.neural_sde.M(t, s).sum(dim=0)

            derivative_M_0 = functorch.jacrev(sum_M, argnums=1)
            derivative_M = lambda t, s: torch.transpose(
                torch.transpose(derivative_M_0(t, s), 1, 2), 0, 1
            )

            M_evals = torch.zeros(len(self.ts), len(self.ts), d, d).to(
                self.ts.device
            )
            derivative_M_evals = torch.zeros(len(self.ts), len(self.ts), d, d).to(
                self.ts.device
            )

            s_vector = []
            t_vector = []
            for k, t in enumerate(self.ts):
                s_vector.append(
                    torch.linspace(t, self.T, self.num_steps + 1 - k).to(self.ts.device)
                )
                t_vector.append(
                    t * torch.ones(self.num_steps + 1 - k).to(self.ts.device)
                )
                
            s_vector = torch.cat(s_vector)
            t_vector = torch.cat(t_vector)
            
            M_evals_all = self.neural_sde.M(
                t_vector,
                s_vector,
            )
            derivative_M_evals_all = derivative_M(
                t_vector,
                s_vector,
            )
            counter = 0
            for k, t in enumerate(self.ts):
                M_evals[k, k:, :, :] = M_evals_all[
                    counter : (counter + self.num_steps + 1 - k), :, :
                ]
                derivative_M_evals[k, k:, :, :] = derivative_M_evals_all[
                    counter : (counter + self.num_steps + 1 - k), :, :
                ]
                counter += self.num_steps + 1 - k

            least_squares_target_integrand_term_1 = torch.einsum(
                "ijkl,jml->ijmk",
                M_evals,
                self.neural_sde.nabla_f(self.ts, states),
            )[:, :-1, :, :]


            M_nabla_b_term = torch.einsum(
                "ijkl,jmln->ijmkn",
                M_evals,
                self.neural_sde.nabla_b(self.ts, states),
            ) - derivative_M_evals.unsqueeze(2)
            least_squares_target_integrand_term_2 = -np.sqrt(
                self.lmbd
            ) * torch.einsum(
                "ijmkn,jmn->ijmk",
                M_nabla_b_term[:, :-1, :, :, :],
                torch.einsum("ij,abj->abi", sigma_inverse_transpose, noises),
            )

            least_squares_target_integrand_term_3 = -torch.einsum(
                "ijmkn,jmn->ijmk",
                M_nabla_b_term[:, :-1, :, :, :],
                torch.einsum("ij,abj->abi", sigma_inverse_transpose, controls),
            )

            M_evals_final = M_evals[:, -1, :, :]
            least_squares_target_terminal = torch.einsum(
                "ikl,ml->imk",
                M_evals_final,
                self.neural_sde.nabla_g(states[-1, :, :]),
            )

            dts = self.ts[1:] - self.ts[:-1]
            least_squares_target_integrand_term_1_times_dt = (
                least_squares_target_integrand_term_1
                * dts.unsqueeze(1).unsqueeze(2).unsqueeze(0)
            )
            least_squares_target_integrand_term_2_times_sqrt_dt = (
                least_squares_target_integrand_term_2
                * torch.sqrt(dts).unsqueeze(1).unsqueeze(2)
            )
            least_squares_target_integrand_term_3_times_dt = (
                least_squares_target_integrand_term_3 * dts.unsqueeze(1).unsqueeze(2)
            )

            cumsum_least_squares_term_1 = torch.sum(
                least_squares_target_integrand_term_1_times_dt, dim=1
            )
            cumsum_least_squares_term_2 = torch.sum(
                least_squares_target_integrand_term_2_times_sqrt_dt, dim=1
            )
            cumsum_least_squares_term_3 = torch.sum(
                least_squares_target_integrand_term_3_times_dt, dim=1
            )

            least_squares_target = (
                cumsum_least_squares_term_1
                + cumsum_least_squares_term_2
                + cumsum_least_squares_term_3
                + least_squares_target_terminal
            )

            control_learned =  self.control(self.ts, states)
            control_target = -torch.einsum(
                "ij,...j->...i",
                torch.transpose(self.sigma, 0, 1),
                least_squares_target,
            )
            
            objective = torch.sum(
                (control_learned - control_target) ** 2
                * weight.unsqueeze(0).unsqueeze(2)
            ) / (states.shape[0] * states.shape[1])


        if algorithm == "SOCM_adjoint":
            nabla_f_evals = self.neural_sde.nabla_f(self.ts, states)
            nabla_b_evals = self.neural_sde.nabla_b(self.ts, states)
            nabla_g_evals = self.neural_sde.nabla_g(states[-1, :, :])

            # print(f'nabla_b_evals.shape: {nabla_b_evals.shape}')

            a_vectors = torch.zeros_like(states)
            a = nabla_g_evals
            a_vectors[-1, :, :] = a

            for k in range(1,len(self.ts)):
                # a += self.dt * (nabla_f_evals[-1-k, :, :] + torch.einsum("mkl,ml->mk", nabla_b_evals[-1-k, :, :, :], a))
                a += self.dt * ((nabla_f_evals[-1-k, :, :] + nabla_f_evals[-k, :, :]) / 2 + torch.einsum("mkl,ml->mk", (nabla_b_evals[-1-k, :, :, :] + nabla_b_evals[-k, :, :, :]) / 2, a))
                a_vectors[-1-k, :, :] = a

            control_learned = self.control(self.ts, states)
            control_target = -torch.einsum(
                "ij,...j->...i",
                torch.transpose(self.sigma, 0, 1),
                a_vectors,
            )
            objective = torch.sum(
                (control_learned - control_target) ** 2
                * weight.unsqueeze(0).unsqueeze(2)
            ) / (states.shape[0] * states.shape[1])

        elif algorithm == "cross_entropy":
            learned_controls =  self.control(self.ts, states)
            integrand_term_1 = -(1 / self.lmbd) * torch.sum(
                learned_controls[:-1, :, :] * controls, dim=2
            )
            integrand_term_2 = (1 / (2 * self.lmbd)) * torch.sum(
                learned_controls**2, dim=2
            )[:-1, :]
            deterministic_integrand = integrand_term_1 + integrand_term_2
            stochastic_integrand = -np.sqrt(1 / self.lmbd) * torch.sum(
                learned_controls[:-1, :, :] * noises, dim=2
            )

            dts = self.ts[1:] - self.ts[:-1]
            deterministic_integrand_times_dt = (
                deterministic_integrand * dts.unsqueeze(1)
            )
            stochastic_integrand_times_sqrt_dt = stochastic_integrand * torch.sqrt(
                dts
            ).unsqueeze(1)

            deterministic_term = torch.sum(deterministic_integrand_times_dt, dim=0)
            stochastic_term = torch.sum(stochastic_integrand_times_sqrt_dt, dim=0)

            objective = torch.mean((deterministic_term + stochastic_term) * weight)

        elif (
            algorithm == "variance"
            or algorithm == "log-variance"
            or algorithm == "moment"
        ):
            learned_controls = self.control(self.ts, states)

            integrand_term_1 = -(1 / self.lmbd) * torch.sum(
                learned_controls[:-1, :, :] * controls, dim=2
            )
            integrand_term_2 = (1 / (2 * self.lmbd)) * torch.sum(
                learned_controls**2, dim=2
            )[:-1, :]
            integrand_term_3 = (
                -(1 / self.lmbd) * self.neural_sde.f(self.ts[0], states)[:-1, :]
            )
            deterministic_integrand = (
                integrand_term_1 + integrand_term_2 + integrand_term_3
            )
            stochastic_integrand = -np.sqrt(1 / self.lmbd) * torch.sum(
                learned_controls[:-1, :, :] * noises, dim=2
            )
            dts = self.ts[1:] - self.ts[:-1]
            deterministic_integrand_times_dt = (
                deterministic_integrand * dts.unsqueeze(1)
            )
            stochastic_integrand_times_sqrt_dt = stochastic_integrand * torch.sqrt(
                dts
            ).unsqueeze(1)

            deterministic_term = torch.sum(deterministic_integrand_times_dt, dim=0)
            stochastic_term = torch.sum(stochastic_integrand_times_sqrt_dt, dim=0)
            g_term = -(1 / self.lmbd) * self.neural_sde.g(states[-1, :, :])
            if algorithm == "log-variance":
                sum_terms = deterministic_term + stochastic_term + g_term
            elif algorithm == "variance":
                sum_terms = torch.exp(deterministic_term + stochastic_term + g_term)
            elif algorithm == "moment":
                sum_terms = deterministic_term + stochastic_term + g_term + self.y0

            weight_2 = torch.ones_like(weight)
            if algorithm == "log-variance" or algorithm == "variance":
                objective = (
                    len(sum_terms)
                    / (len(sum_terms) - 1)
                    * (
                        torch.mean(sum_terms**2 * weight_2)
                        - torch.mean(sum_terms * weight_2) ** 2
                    )
                )
            elif algorithm == "moment":
                objective = torch.mean(sum_terms**2 * weight_2)
        
        if algorithm == "adjoint_matching":            
            detached_states = states.detach()
            nabla_f_evals = self.neural_sde.nabla_f(self.ts, detached_states)
            nabla_b_evals = self.neural_sde.nabla_b(self.ts, detached_states)
            nabla_g_evals = self.neural_sde.nabla_g(detached_states[-1, :, :])

            a_vectors = torch.zeros_like(states)
            a = nabla_g_evals
            a_vectors[-1, :, :] = a

            for k in range(1,len(self.ts)):
                a += self.dt * ((nabla_f_evals[-1-k, :, :] + nabla_f_evals[-k, :, :]) / 2 + torch.einsum("mkl,ml->mk", (nabla_b_evals[-1-k, :, :, :] + nabla_b_evals[-k, :, :, :]) / 2, a))
                a_vectors[-1-k, :, :] = a

            control_learned = self.control(self.ts, detached_states)
            control_target = -torch.einsum(
                "ij,...j->...i",
                torch.transpose(self.sigma, 0, 1),
                a_vectors,
            )

            objective = torch.sum(
                (control_learned - control_target) ** 2
            ) / (states.shape[0] * states.shape[1])

        if verbose:
            # To print amount of memory used in GPU
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print("Total memory:", info.total / 1048576, "MiB")
            print("Free memory:", info.free / 1048576, "MiB")
            print("Used memory:", info.used / 1048576, "MiB")
            nvidia_smi.nvmlShutdown()

        return (
            objective,
            torch.logsumexp(log_weight + log_normalization_const, dim=0) - np.log(log_weight.shape[0]),
            torch.std(weight),
        )

    """
    Loss for deep FBSDE method
    """
    def fbsde_loss(
        self,
        reg = 1.0,
        state0=None,
        verbose = False,
        ):

        device   = self.x0.device
        N   = self.ts.shape[0] - 1
        dts = self.ts[1:] - self.ts[:-1]
        sqrt_dts = dts.sqrt()
        Mbatch = state0.shape[0] // 2

        (states, noises,
        _det, _sto, _ter,
        Z_all) = stochastic_trajectories(
            self.neural_sde,
            state0,
            self.ts.to(device),
            self.lmbd,
            detach=False,
        )

        noises = noises * sqrt_dts[:,None,None]
        
        X_N     = states[-1]                 
        g_X_N   = self.neural_sde.g(X_N)

        f_all = self.neural_sde.f(self.ts, states)[:-1] + 1/2 * Z_all.norm(dim=-1)**2

        z_dot_dw = (Z_all * noises).sum(-1)

        int_fh     = (f_all * dts.unsqueeze(1)).sum(0)
        sum_z_dw   = z_dot_dw.sum(0)

        Y0_first  = g_X_N[:Mbatch] + int_fh[:Mbatch] - sum_z_dw[:Mbatch]

        Y0_init = g_X_N[:Mbatch].mean()         

        f_2    = f_all[:, Mbatch:]           
        z_dw_2 = z_dot_dw[:, Mbatch:]         
        g_2    = g_X_N[Mbatch:]

        Y_N_2  = Y0_init - (f_2 * dts.unsqueeze(1)).sum(0) + z_dw_2.sum(0)

        term_error_sq = (g_2 - Y_N_2).pow(2)

        # ------------------------------------------------------------------
        #  robust FBSDE loss  L = 1/Mbatch ( Σ Y0_first  + λ Σ |g - Y_N|² )
        # ------------------------------------------------------------------
        loss = (Y0_first.sum() + reg * term_error_sq.sum()) / Mbatch

        if verbose:
            print(
                f"[FBSDE]  E[Y0] = {Y0_first.mean().item(): .4e}   "
                f"Var term = {(term_error_sq.mean()).item(): .4e}   "
                f"loss = {loss.item(): .4e}"
            )

        return loss