"""

The main code for the solver modules. The NeuralSDE API and code for the IDO methods are adapted from https://github.com/facebookresearch/SOC-matching.

"""

import torch
import torch.nn as nn
import numpy as np
import nvidia_smi
import functorch

from SOC_eigf.models import FullyConnectedUNet, SIREN, GaussianNet, SigmoidMLP
from SOC_eigf.utils import mala_samples

class NeuralSDE(nn.Module):
    """
    Class that contains the models and SDE parameters.

    eigf_cfg:
        k: number of eigenfunctions to compute
        hdims: hidden layer dimensions for eigenfunction model
        arch: activation function to use {'SIREN', 'GAUSS'}

    ido_cfg:
        hdims: hidden layer dimensions for control
        hdims_M: hidden layer dimensions for M matrix (for SOCM method)
        gamma{,2,3}: gamma parameters
        scaling_factor_nabla_V: init scaling for nabla_V
        scaling_factor_M: init scaling for M
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
        ido_cfg=None
    ):
        super().__init__()
        assert method in ['EIGF','IDO','COMBINED']
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

        self.hdims_ido = ido_cfg.get('hdims',[256,128,64])
        self.hdims_M = ido_cfg.get('hdims_M',[128,128])
        self.scaling_factor_nabla_V = ido_cfg.get('scaling_factor_nabla_V', 1.0)
        self.scaling_factor_M= ido_cfg.get('scaling_factor_M', 1.0)
        self.gamma = ido_cfg.get('gamma',1.0)
        self.gamma2 = ido_cfg.get('gamma2',1.0)
        self.gamma3 = ido_cfg.get('gamma3',1.0)

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
        network_modules = {'SIREN': SIREN, 'GAUSS': GaussianNet}

        if self.method == "EIGF":

            # ground state model
            self.eigf_gs_model = network_modules[self.arch_eigf](
                    dim=self.dim,
                    hdims=self.hdims_eigf,
                    k = 1
                    ).to(self.device)
            
            if self.k > 1:
                self.eigf_model = nn.ModuleList([
                        network_modules[self.arch_eigf](
                        dim=self.dim,
                        hdims=self.hdims_eigf,
                        k = self.k-1
                        ).to(self.device) for i in range(self.k)
                    ])

            # save eigvals
            self.register_buffer('eigvals', torch.zeros(self.k, device=self.device))
            self.register_buffer('inner_prods',torch.ones(self.k, device=self.device))
            self.register_buffer('norms', torch.ones(self.k, device=self.device))

            def gs_model_fn(x):
                out = self.eigf_gs_model(x)
                return out, out
            
            self.eigf_gs_model_jac = torch.vmap(torch.func.jacrev(gs_model_fn,has_aux=True))
            self.eigf_gs_model_hessian = torch.func.hessian(lambda x: gs_model_fn(x)[0])
            self.eigf_gs_model_laplacian = torch.vmap(lambda x: self.eigf_gs_model_hessian(x).diagonal(dim1=-1,dim2=-2).sum(dim=-1))
        
        elif self.method == 'IDO':
            self.ido_model = FullyConnectedUNet(
                dim=self.dim,
                hdims=self.hdims_ido,
                scaling_factor=self.scaling_factor_nabla_V,
            ).to(self.device)

            self.gamma = torch.nn.Parameter(torch.tensor([self.gamma]).to(self.device))
            self.M = SigmoidMLP(
                dim=self.dim,
                hdims=self.hdims_M,
                gamma=self.gamma,
                scaling_factor=self.scaling_factor_M,
            ).to(self.device)

        elif self.method == 'COMBINED':
            raise NotImplementedError
        
    def nabla_V(self,t,x):
        """
        Output nabla_V estimate.
        """
        t = t.reshape(-1)
        t_rev = self.T - t

        if not self.confining:
            # f = exp(E) f_theta, so grad log f = E + grad log f_theta
            shift = - self.b(None, x) * 2 / self.lmbd
        else:
            shift = 0
        
        if len(x.shape) == 2:
            if self.method == "EIGF":
                Dfx, fx = self.eigf_gs_model_jac(x)
                Dfx = Dfx / self.norms[None,:,None]

                return -self.lmbd * (Dfx[:,0,:] + shift)
        
        if len(x.shape) == 3:
            x_reshaped = x.reshape(-1,self.dim)

            if self.method == "EIGF":
                Dfx, fx = self.eigf_gs_model_jac(x_reshaped)
                Dfx = Dfx / self.norms[None,:,None]
                Dfx = torch.reshape(Dfx, (x.shape[0],x.shape[1], self.k, x.shape[2]))

                return -self.lmbd * (Dfx[:,:,0,:] + shift)
            

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
        self.dim = neural_sde.dim
        self.x0 = x0
        self.ut = ut
        self.T = T
        self.ts = torch.linspace(0,T, num_steps+1)
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

        self.neural_sde = neural_sde

    def control(self, t0, x0, k = 1):
        x0 = x0.reshape(-1, self.dim)
        t0_expanded = t0.expand(x0.shape[0])
        nabla_V = self.neural_sde.nabla_V(t0_expanded, x0)
        learned_control = -torch.einsum(
            "ij,bj->bi", torch.transpose(self.sigma, 0, 1), nabla_V
        )
        return learned_control
    
    # ground state loss
    def gs_loss(
        self,
        verbose=False
    ):
        if self.samples is None:
            if verbose:
                print('Burning in Langevin...')
            self.samples = self.x0.repeat(self.nsamples,1)

            burnin_langevin_ts = torch.linspace(0,self.langevin_dt * self.langevin_burnin_steps, self.langevin_burnin_steps+1).to(self.x0.device)
            self.sample_langevin_ts = torch.linspace(0,self.langevin_dt * self.langevin_sample_steps, self.langevin_sample_steps+1).to(self.x0.device)
            self.samples = mala_samples(self.neural_sde,self.samples,burnin_langevin_ts,self.lmbd,verbose=verbose)

            shift = self.samples.mean(dim=0)
            scale = self.samples.std(dim=0)
            
            if verbose:
                print(f'Completed burn-in. Samples mean {shift}, std {scale}')
            
            with torch.no_grad():
                self.neural_sde.eigf_gs_model.shift.copy_(shift)
                self.neural_sde.eigf_gs_model.scale.copy_(scale)
        
        else:
            self.samples = mala_samples(self.neural_sde,self.samples,self.sample_langevin_ts,self.lmbd)

        Dfx, fx = self.neural_sde.eigf_gs_model_jac(self.samples)

        Rx = self.neural_sde.f(None,self.samples) * 2 / self.lmbd**2

        if self.neural_sde.confining:
            grad_norm = torch.norm(Dfx,dim=2,p=2).clip(min=1e-4)
        else:
            grad_Ex = - self.neural_sde.b(None, self.samples) * 2 / self.lmbd
            grad_norm = torch.norm(Dfx[:,0,:] + grad_Ex,dim=1,p=2).clip(min=1e-4)

        sq_f_norm = torch.logsumexp(fx[:,None,:] + fx[:,:,None], dim = 0).exp() / fx.shape[0]
        orth_loss = (sq_f_norm - 1)**2

        if self.eigf_loss in ['var','ritz']:
            # <\nabla f, \nabla f>_mu
            sq_grad_norm = torch.logsumexp(2*fx[:,0] + 2*grad_norm[:,0].log(),dim=0).exp() / fx.shape[0]

            Rx = torch.clip(torch.abs(Rx),min=1e-8) * torch.sign(Rx)
            R_pos_idx = Rx.squeeze() > 0
            
            # <f, Rf>_mu
            R_norm = (torch.logsumexp(2*fx[R_pos_idx,0] + Rx[R_pos_idx].log(),dim=0).exp() - torch.logsumexp(2*fx[~R_pos_idx,0] + (-Rx[~R_pos_idx]).log(),dim=0).exp()) / fx.shape[0]

            var_loss = sq_grad_norm + R_norm

            if self.eigf_loss == 'var':
                self.neural_sde.eigvals[0] = 2/self.beta * (1-sq_f_norm.detach())
                return self.beta*var_loss + orth_loss, var_loss, orth_loss, fx, Dfx
            
            elif self.eigf_loss == "ritz":
                self.neural_sde.eigvals[0] = (var_loss / sq_f_norm).detach()
                return var_loss / sq_f_norm + orth_loss, var_loss / sq_f_norm, orth_loss, fx, Dfx
            
        else:
            Deltafx = self.neural_sde.eigf_gs_model_laplacian(self.samples)
            grad_E = -self.neural_sde.b(0, self.samples) * 2 / self.neural_sde.lmbd
            Lfx = -Deltafx - grad_norm**2 + torch.einsum('ij,ikj->ik',grad_E,Dfx) + Rx.unsqueeze(1)

            sq_diff = ((Lfx - self.neural_sde.eigvals[0])**2).clip(min=1e-5)

            if self.eigf_loss == 'rel':
                rel_loss = sq_diff.mean(dim=0)
                return rel_loss + orth_loss, rel_loss, orth_loss, fx, Dfx
            
            elif self.eigf_loss == "pinn":
                pinn_loss = torch.logsumexp(sq_diff.log() + 2*fx,dim=0).exp() / fx.shape[0]
                return pinn_loss / sq_f_norm + orth_loss, pinn_loss, orth_loss, fx, Dfx
            
