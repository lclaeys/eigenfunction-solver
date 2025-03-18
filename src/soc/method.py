import torch
import torch.nn as nn
import numpy as np
import nvidia_smi
import functorch

from src.soc.models import FullyConnectedUNet, FullyConnectedUNet2
from src.soc.utils import langevin_samples, mala_samples

from socmatching.SOC_matching import utils, models
from socmatching.SOC_matching.method import NeuralSDE, SOC_Solver

class EigenSDE(NeuralSDE):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
            self,
            device="cuda",
            dim=2,
            hdims=[256, 128, 64],
            prior=None,
            joint=True,
            u=None,
            lmbd=1.0,
            sigma=torch.eye(2),
            T=1.0,
            k=1
    ):
        super().__init__(
            device=device,
            dim=dim,
            hdims=hdims,
            u=u,
            lmbd=lmbd,
            sigma=sigma,
            T=T
        )

        self.k = k
        self.prior = prior
        self.joint = joint

    # Control
    def control(self, t, x, k = None, reg = 5e-3, verbose=False):
        if verbose:
            print(
                f"self.use_learned_control: {self.use_learned_control}, self.u: {self.u}"
            )
        if self.use_learned_control:
            if len(x.shape) == 2:
   
                learned_control = -torch.einsum(
                    "ij,bj->bi",
                    torch.transpose(self.sigma, 0, 1),
                    self.nabla_V(t,x,k,reg),
                )

                return learned_control
            
            # x.shape = (N,B,d)
            if len(x.shape) == 3:
                learned_control = -torch.einsum(
                    "ij,abj->abi",
                    torch.transpose(self.sigma, 0, 1),
                    self.nabla_V(t,x,k,reg),
                )
                return learned_control
        else:
            if self.u is None:
                return None
            else:
                return self.u(t, x)

    def initialize_models(self):
        init_scale = torch.ones(self.k)
        if self.prior == "positive":
            init_scale[0] = 1e-2

        if self.joint:
            self.eigf_model = FullyConnectedUNet(
                dim=self.dim,
                hdims=self.hdims,
                k = self.k
            ).to(self.device)
        else:
            self.eigf_models = nn.ModuleList([
                FullyConnectedUNet(
                dim=self.dim,
                hdims=self.hdims,
                k = 1,
                scaling_factor=init_scale[i]
                ).to(self.device) for i in range(self.k)
            ])
            print(len(self.eigf_models))

        # Use learned control in the stochastic_trajectories function
        self.use_learned_control = True
        self.register_buffer('eigvals', torch.zeros(self.k, device=self.device))
        self.register_buffer('inner_prods',torch.ones(self.k, device=self.device))

        if self.joint:
            self.transformation = torch.eye(self.k, device=self.device)
            self.indices = torch.arange(self.k)
        
        self.register_buffer('norms', torch.ones(self.k, device=self.device))

        if self.joint:
            if not self.training:
                def model_fn(x):
                    out = (self.eigf_model(x) @ self.transformation)[self.indices]
                    return out, out
            else:
                def model_fn(x):
                    out = self.eigf_model(x)
                    return out, out
        else:
            def model_fn(x):
                outs = [model(x).squeeze(0) for model in self.eigf_models]
                out = torch.stack(outs,dim = 0)
                return out, out
        
        self.model_jac = torch.vmap(torch.func.jacrev(model_fn,has_aux=True))

    def nabla_V(self, t, x, k = None, reg = 5e-3):
        """
        Compute nabla_V using first k eigenfunctions
        Args:
           t (tensor)[N]: times
           x (tensor)[N,d] or [N,B,d]: positions 
        Returns:
            nabla_V (tensor)[N,d] or [N,B,d]: gradients

        When training, possibly perform PCA.
        """
        if k is None:
            k = self.k
        
        t = t.reshape(-1)
        t_rev = self.T - t

        regularizer = self.inner_prods[0] * reg

        if not self.confining:
            # f = exp(E) f_theta, so grad log f = E + grad log f_theta
            shift = - self.b(None, x) * 2 / self.lmbd
        else:
            shift = 0
        
        if len(x.shape) == 2:
            # (N,k,d) and (N,k)
            Dfx, fx = self.model_jac(x)
            Dfx = Dfx / self.norms[None,:,None]
            fx = fx / self.norms[None,:]

            if self.prior == "positive":
                if k == 1:
                    return - self.lmbd * (Dfx[:,0,:] + shift)
                else:
                    fx[:,0] = fx[:,0].exp()
                    Dfx[:,0,:] = Dfx[:,0,:] * fx[:,0,None]
            
            # (N)
            v = torch.sum(self.inner_prods[None,:k]
                          * torch.exp(-t_rev[:,None] * self.eigvals[None,:k] * self.lmbd / 2)
                          * fx[:,:k], dim = 1)
            
            # (N,d)
            nabla_v = torch.sum(self.inner_prods[None,:k,None]
                          * torch.exp(-t_rev[:,None,None] * self.eigvals[None,:k,None] * self.lmbd / 2)
                          * Dfx[:,:k,:], dim = 1)
                        
            v = torch.sign(v) * (torch.abs(v) + regularizer)
            
            return -self.lmbd *  (nabla_v / v[:,None] + shift)
        
        if len(x.shape) == 3:
            x_reshaped = x.reshape(-1,self.dim)
            Dfx, fx = self.model_jac(x_reshaped)

            Dfx = Dfx / self.norms[None,:,None]
            fx = fx / self.norms[None,:]
            
            Dfx = torch.reshape(Dfx, (x.shape[0],x.shape[1], self.k, x.shape[2]))
            fx = torch.reshape(fx, (x.shape[0],x.shape[1], self.k))

            if self.prior == "positive":
                if k == 1:
                    return - self.lmbd * (Dfx[:,:,0,:] + shift)
                else:
                    fx[:,:,0] = fx[:,:,0].exp()
                    Dfx[:,:,0,:] = Dfx[:,:,0,:] * fx[:,:,0,None]

            # (N,B)
            v = torch.sum(self.inner_prods[None,None,:k]
                          * torch.exp(-t_rev[:,None,None] * self.eigvals[None,None,:k] * self.lmbd / 2)
                          * fx[:,:,:k], dim = 2)
            
            # (N,B,d)
            nabla_v = torch.sum(self.inner_prods[None,None,:k,None]
                          * torch.exp(-t_rev[:,None,None,None] * self.eigvals[None,None,:k,None] * self.lmbd / 2)
                          * Dfx[:,:,:k,:], dim = 2)

            v = torch.sign(v) * (torch.abs(v) + regularizer)
            
            return -self.lmbd * (nabla_v / v[:,:,None] + shift)
        
    def compute_inner_prods(self, 
                            samples):
        if self.joint:
            fx = (self.eigf_model(samples) @ self.transformation)[:,self.indices]
        else:
            fx = torch.stack([model(samples).squeeze(1) for model in self.eigf_models],dim = 1)

        if self.confining:
            gx = - self.g(samples) / self.lmbd
        else:
            gx = - (self.g(samples) - 2*self.energy(samples)) / self.lmbd

        inner_prods = torch.mean(fx * gx.exp().unsqueeze(1),dim=0)
        
        if self.prior == "positive":
            inner_prods[0] = torch.logsumexp(fx[:,0] + gx,dim=0).exp() / fx.shape[0]

        inner_prods = inner_prods / self.norms
        self.inner_prods = inner_prods
        print(self.inner_prods)

class EigenSolver(SOC_Solver):
    noise_type = "diagonal"
    sde_type = "ito"

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
        langevin_burnin_steps = 1000,
        langevin_sample_steps = 100,
        langevin_dt = 0.01,
        beta = 1.0,
    ):
        super().__init__(
            neural_sde,
            x0,
            ut
        )
        self.dim = neural_sde.dim
        self.neural_sde = neural_sde
        self.x0 = x0
        self.ut = ut
        self.T = T
        self.ts = torch.linspace(0, T, num_steps + 1).to(x0.device)
        self.num_steps = num_steps
        self.dt = T / num_steps
        self.lmbd = lmbd
        self.d = d
        self.y0 = torch.nn.Parameter(torch.randn(1, device=x0.device))
        self.sigma = sigma
        self.samples = None
        self.langevin_burnin_steps = langevin_burnin_steps
        self.langevin_sample_steps = langevin_sample_steps
        self.langevin_dt = langevin_dt
        self.f_scaling = torch.ones(neural_sde.k, device=x0.device,requires_grad=False)
        self.beta = torch.ones(neural_sde.k, device=x0.device,requires_grad=False) * beta
    
    def control(self, t0, x0, k = 1):
        x0 = x0.reshape(-1, self.dim)
        t0_expanded = t0.expand(x0.shape[0])
        nabla_V = self.neural_sde.nabla_V(t0_expanded, x0, k = k)
        learned_control = -torch.einsum(
            "ij,bj->bi", torch.transpose(self.sigma, 0, 1), nabla_V
        )
        return learned_control

    def loss(
        self,
        sample_size=65536,
        verbose=False,
        compute_ab=False,
        update_beta=False
    ):
        if self.samples is None:
            if verbose:
                print('Burning in Langevin...')
            self.samples = self.x0.repeat(sample_size,1)

            burnin_langevin_ts = torch.linspace(0,self.langevin_dt * self.langevin_burnin_steps, self.langevin_burnin_steps+1).to(self.x0.device)
            self.sample_langevin_ts = torch.linspace(0,self.langevin_dt * self.langevin_sample_steps, self.langevin_sample_steps+1).to(self.x0.device)
            self.samples = mala_samples(self.neural_sde,self.samples,burnin_langevin_ts,self.lmbd)
        
        else:
            self.samples = mala_samples(self.neural_sde,self.samples,self.sample_langevin_ts,self.lmbd)
        
        Dfx, fx = self.neural_sde.model_jac(self.samples)

        fx = fx * self.f_scaling[None,:].clone()
        Dfx = Dfx * self.f_scaling[None,:,None].clone()

        Rx = self.neural_sde.f(None,self.samples) * 2 / self.lmbd**2
        
        # (k,)
        if self.neural_sde.confining:
            grad_norm = torch.norm(Dfx,dim=2,p=2)
            sq_grad_norms = torch.mean(grad_norm**2,dim=0)
            
            if self.neural_sde.prior == "positive":
                sq_grad_norms[0] = torch.logsumexp(2*fx[:,0] + 2*grad_norm[:,0].log(),dim=0).exp() / fx.shape[0]
        else:
            grad_Ex = - self.neural_sde.b(None, self.samples) * 2 / self.lmbd
            sq_grad_norms = torch.mean(torch.norm(Dfx + fx[:,:,None]*grad_Ex[:,None,:],dim=2,p=2)**2,dim=0)

            if self.neural_sde.prior == "positive":
                grad_norm = torch.norm(Dfx[:,0,:] + grad_Ex,dim=1,p=2)
                sq_grad_norms[0] = torch.logsumexp(2*fx[:,0] + 2*grad_norm.log(),dim=0).exp() / fx.shape[0]

        R_norms = torch.mean(fx**2 * Rx[:,None],dim=0)
        if self.neural_sde.prior == "positive":
            R_norms[0] = torch.logsumexp(2*fx[:,0] + Rx.log(),dim=0).exp() / fx.shape[0]

        if self.neural_sde.joint:
            var_loss = torch.sum(sq_grad_norms + R_norms)
            
            fx_cov = torch.mean(fx[:,None,:] * fx[:,:,None], dim = 0)
            orth_loss = ((fx_cov - torch.eye(fx.shape[1],device=fx.device))**2).sum()
        
        else:
            var_loss = sq_grad_norms + R_norms
            fx_cov = torch.mean(fx[:,None,:] * fx[:,:,None], dim = 0)
            
            if self.neural_sde.prior == "positive":
                fx_cov[0,0] = torch.logsumexp(fx[:,None,0] + fx[:,0,None],dim=0).exp() / fx.shape[0]

            orth_loss = ((fx_cov - torch.eye(fx.shape[1],device=fx.device))**2).cumsum(dim=0).cumsum(dim=1).diagonal()
        
        fx_cov = fx_cov.detach()
        self.neural_sde.eigvals = 2 / self.beta * (1 - fx_cov.diagonal())
        
        loss = self.beta.clone() * var_loss + orth_loss, var_loss, orth_loss
        
        if update_beta:
            new_beta = 1 / torch.clip(torch.abs(self.neural_sde.eigvals),min=1e-3)
            self.f_scaling *= torch.sqrt((2+new_beta*self.neural_sde.eigvals) / (2+self.beta*self.neural_sde.eigvals))
            self.beta = new_beta

        return loss
    
    def compute_eigvals(self, 
                        beta=1.0,
                        pca_reg=1e-6):
        if self.neural_sde.joint:
            fx = self.neural_sde.eigf_model(self.samples).to('cpu')
            fx = fx.double()

            cov = (fx.T @ fx) / fx.shape[0]

            error = torch.linalg.eigvalsh(cov)[0]
            if error < 0:
                pca_reg += -error*1.1

            cov = cov + pca_reg*torch.eye(cov.shape[0],dtype=torch.float64)
            D, U = torch.linalg.eigh(cov)
            D, U = D.float().to(self.neural_sde.device), U.float().to(self.neural_sde.device)

            eigvals = 2/beta*(1-D)
            self.neural_sde.transformation = U @ torch.diag(D**(-1/2))
            self.neural_sde.indices = torch.argsort(eigvals)
            self.neural_sde.eigvals = eigvals[self.neural_sde.indices]
            self.neural_sde.eigvals = self.neural_sde.eigvals - self.neural_sde.eigvals[0]
        
        else:
            fx = torch.stack([model(self.samples).squeeze(1) for model in self.neural_sde.eigf_models], dim = 1).detach()

            norms = torch.mean(fx**2, dim = 0).sqrt()
            if self.neural_sde.prior == "positive":
                norms[0] = (torch.logsumexp(2*fx[:,0],dim=0).exp() / fx.shape[0]).sqrt()
            
            self.neural_sde.norms = 0.95 * self.neural_sde.norms + 0.05 * norms
            self.neural_sde.eigvals = 2/beta * (1 - self.neural_sde.norms**2)
            print(self.neural_sde.eigvals)
            self.neural_sde.eigvals = self.neural_sde.eigvals - self.neural_sde.eigvals[0]

class CombinedSDE(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
            self,
            neural_sde,
            eigen_sde,
            cutoff_T = 1.0,
            train_value=False,
            use_terminal=False
    ):
        super().__init__()
        self.neural_sde = neural_sde
        self.eigen_sde = eigen_sde
        self.cutoff_T = cutoff_T
        self.T = self.neural_sde.T
        self.lmbd = self.neural_sde.lmbd
        self.dim = self.neural_sde.dim
        self.device = self.neural_sde.device
        self.confining = self.eigen_sde.confining
        self.b = self.eigen_sde.b
        self.nabla_b = self.eigen_sde.nabla_b
        self.f = self.eigen_sde.f
        self.nabla_f = self.eigen_sde.nabla_f
        self.g = self.eigen_sde.g
        self.nabla_g = self.eigen_sde.nabla_g
        self.energy = self.eigen_sde.energy
        self.use_learned_control = True
        self.sigma = self.eigen_sde.sigma
        self.gamma = self.neural_sde.gamma
        self.gamma2 = self.neural_sde.gamma2
        self.gamma3 = self.neural_sde.gamma3
        self.lmbd_1 = nn.Parameter(torch.tensor([self.eigen_sde.eigvals[1] - self.eigen_sde.eigvals[0]],device=self.device) * self.lmbd / 2,requires_grad=False) 
        self.train_value = train_value
        self.use_terminal = use_terminal
        if self.use_terminal:
            self.neural_sde.terminal_decay = nn.Parameter(torch.tensor([self.eigen_sde.eigvals[1] - self.eigen_sde.eigvals[0]],device=self.device) * self.lmbd / 2,requires_grad=True)


    def initialize_models(self):
        self.neural_sde.initialize_models()

        if self.train_value:
            self.neural_sde.model = FullyConnectedUNet(
                    dim=self.dim+1,
                    hdims=self.neural_sde.hdims,
                    k = 1,
                    scaling_factor = self.neural_sde.scaling_factor_nabla_V
                ).to(self.device)
            self.neural_sde.nabla_V = None

        else:
            self.neural_sde.model = FullyConnectedUNet2(
                    dim=self.dim,
                    hdims=self.neural_sde.hdims,
                    scaling_factor = self.neural_sde.scaling_factor_nabla_V
                ).to(self.device)
            
            self.neural_sde.nabla_V = None
        
        self.M = self.neural_sde.M
    
        def eigf_fn(x):
            out = self.eigen_sde.eigf_models[0](x)[0]
            return out, out
        
        self.eigf_jac = torch.vmap(torch.func.jacrev(eigf_fn,has_aux=True))
        self.epsilon_param = nn.Parameter(torch.tensor([-5.0],device=self.device))
        self.epsilon = torch.log(1 + torch.exp(self.epsilon_param))
        
        def model_fn(x):
            out = self.neural_sde.model(x)[0]
            return out, out
        
        self.model_jac = torch.vmap(torch.func.jacrev(model_fn,has_aux=True))
        self.neural_sde.gamma.requires_grad_(False)
    
    def control(self, t, x, verbose=False):
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
    
    def nabla_V(self, t, x):
        """
        Compute nabla_V by combining eigenfunctions and neural network
        """
        k = self.eigen_sde.k
        
        if torch.all(t < self.T - self.cutoff_T):
            eigf_only = True
        else:
            eigf_only = False

            if len(t.shape) >= 1:
                model_idx = torch.searchsorted(t,self.T - self.cutoff_T) - 1 # find cutoff index
                model_t = t[model_idx:]
                model_x = x[model_idx:]
                model_t_rev = self.T - model_t
            else:
                model_idx = 0
                model_t = t
                model_x = x
                model_t_rev = (self.T - model_t).unsqueeze(0)

        if not self.confining:
            # f = exp(E) f_theta, so grad log f = grad E + grad log f_theta
            shift = - self.b(None, x) * 2 / self.lmbd
        else:
            shift = 0
        
        if len(x.shape) == 2:
            x = x.reshape(-1,self.dim)

            Dphix, phix = self.eigf_jac(x)

            # if not self.confining:
            #     grad_E = - self.b(None, x) * 2 / self.lmbd
            #     E = self.energy(x) * 2 / self.lmbd
            #     Dphix = (Dphix + grad_E)
            #     phix = phix
                        
            self.epsilon = torch.log(1 + torch.exp(self.epsilon_param))
            phix = torch.clamp(phix, min = self.epsilon)

            u_eigf = - self.lmbd * (Dphix / phix[:,None] + shift)
            u = u_eigf

            if not eigf_only:
                if self.train_value:
                    model_ts_expand = model_t_rev.reshape(-1, 1).expand(model_x.shape[0], 1)
                    model_tx = torch.cat([model_ts_expand, model_x], dim=-1)

                    Dfx, fx = self.model_jac(model_tx)
                    Dfx = Dfx[:,1:]
                
                    v_mod =  torch.exp(self.lmbd_1 * model_t_rev) + fx
                    v_mod = torch.clamp(v_mod,min=self.epsilon)
                
                    u_mod = - self.lmbd * (Dfx / v_mod[:,None])
                
                    u = torch.cat([u_eigf[:model_idx], u_mod + u_eigf[model_idx:]], dim=0)
                
                else:
                    model_ts_expand = model_t_rev.reshape(-1, 1).expand(model_x.shape[0], 1)
                    model_ttx = torch.cat([model_ts_expand, torch.exp(- model_ts_expand * self.lmbd_1), model_x], dim=-1)

                    u_mod = self.neural_sde.model(model_ttx)

                    u = torch.cat([u_eigf[:model_idx], u_mod*torch.exp(- model_ts_expand * self.lmbd_1) + u_eigf[model_idx:]], dim=0)
            
            if self.use_terminal:

                terminal_weight = torch.exp(-(self.T - t)*self.neural_sde.terminal_decay).unsqueeze(1)
                u = (1-terminal_weight) * u + terminal_weight * self.nabla_g(x)

            return u

        if len(x.shape) == 3:
            x_reshaped = x.reshape(-1,self.dim)

            Dphix, phix = self.eigf_jac(x_reshaped)

            Dphix = torch.reshape(Dphix, (x.shape[0],x.shape[1], x.shape[2]))
            phix = torch.reshape(phix, (x.shape[0],x.shape[1]))

            # if not self.confining:
            #     grad_E = - self.b(None, x) * 2 / self.lmbd
            #     E = self.energy(x) * 2 / self.lmbd
            #     Dphix = (Dphix + grad_E)
            #     phix = phix
            
            self.epsilon = torch.log(1 + torch.exp(self.epsilon_param))
            phix = torch.clamp(phix, min = self.epsilon)
            
            u_eigf = - self.lmbd * (Dphix / phix[:,:,None] + shift)

            u = u_eigf

            if not eigf_only:
                if self.train_value:
                    model_ts_repeat = model_t_rev.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], 1)
                    model_tx = torch.cat([model_ts_repeat, model_x], dim=-1)
                    model_tx_reshape = torch.reshape(model_tx, (-1, model_tx.shape[2]))

                    Dfx, fx = self.model_jac(model_tx_reshape)

                    Dfx = torch.reshape(Dfx[:,1:], (model_x.shape[0],model_x.shape[1], model_x.shape[2]))
                    fx = torch.reshape(fx, (model_x.shape[0],model_x.shape[1]))

                    v_mod =  torch.exp(self.lmbd_1 * model_t_rev)[:,None] + fx
                    v_mod = torch.clamp(v_mod,min=self.epsilon)            
                
                    u_mod = - self.lmbd * (Dfx / v_mod[:,:,None])

                    u = torch.cat([u_eigf[:model_idx], u_mod + u_eigf[model_idx:]], dim=0)
                else:
                    model_ts_repeat = model_t_rev.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], 1)
                    model_ttx = torch.cat([model_ts_repeat, torch.exp(- model_ts_repeat * self.lmbd_1), model_x], dim=-1)
                    model_ttx_reshape = torch.reshape(model_ttx, (-1, model_ttx.shape[2]))

                    fx = self.neural_sde.model(model_ttx_reshape)
                    u_mod = torch.reshape(fx, (model_x.shape[0],model_x.shape[1], model_x.shape[2]))

                    u = torch.cat([u_eigf[:model_idx], u_mod*torch.exp(- model_ts_repeat * self.lmbd_1) + u_eigf[model_idx:]], dim=0)
            
            if self.use_terminal:
                terminal_weight = torch.exp(-(self.T - t)*self.neural_sde.terminal_decay).unsqueeze(1).unsqueeze(2)
                u = (1-terminal_weight) * u + terminal_weight * self.nabla_g(x)

            return u

class CombinedSolver(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
            self,
            combined_sde,
            x0,
            T,
            num_steps,
            lmbd,
            d,
            sigma = torch.eye(2)
    ):  
        super().__init__()
        self.neural_sde = combined_sde
        self.x0 = x0
        self.T = T
        self.ts = torch.linspace(0, T, num_steps + 1).to(x0.device)
        self.num_steps = num_steps
        self.dt = T / num_steps
        self.lmbd = lmbd
        self.d = d
        self.sigma = sigma

    def control(self, t0, x0):
        return self.neural_sde.control(t0, x0)
    
    def loss(
        self,
        batch_size,
        algorithm,
        total_n_samples=65536,
        use_stopping_time=False,
        verbose=False,
        add_weights=False,
        log_normalization_const=0.0,
        state0=None
    ):
        self.num_steps = self.ts.shape[0] - 1
        d = state0.shape[1]
        detach = algorithm != "rel_entropy"
        (
            states,
            noises,
            stop_indicators,
            fractional_timesteps,
            log_path_weight_deterministic,
            log_path_weight_stochastic,
            log_terminal_weight,
            controls,
        ) = utils.stochastic_trajectories(
            self.neural_sde,
            state0,
            self.ts.to(state0),
            self.lmbd,
            detach=detach,
        )

        unsqueezed_stop_indicators = stop_indicators.unsqueeze(2)
        weight = torch.exp(
            - log_normalization_const
            + log_path_weight_deterministic
            + log_path_weight_stochastic
            + log_terminal_weight
        )
        log_weight = log_path_weight_deterministic + log_path_weight_stochastic + log_terminal_weight

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
            identity = torch.eye(d).to(self.x0.device)

            if use_stopping_time:
                sum_M = lambda t, s, stopping_timestep_values: self.neural_sde.M(
                    t, s, stopping_timestep_values
                ).sum(dim=0)

                derivative_M_0 = functorch.jacrev(sum_M, argnums=1)
                derivative_M = lambda t, s, stopping_timestep_values: torch.transpose(
                    torch.transpose(
                        torch.transpose(
                            derivative_M_0(t, s, stopping_timestep_values), 2, 3
                        ),
                        1,
                        2,
                    ),
                    0,
                    1,
                )

                M_evals = torch.zeros(len(self.ts), len(self.ts), batch_size, d, d).to(
                    self.ts.device
                )
                derivative_M_evals = torch.zeros(
                    len(self.ts), len(self.ts), batch_size, d, d
                ).to(self.ts.device)

            else:
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

            if use_stopping_time:
                stopping_function_output_int = (self.neural_sde.Phi(states) > 0).to(
                    torch.int
                )
                stopping_timestep = (
                    torch.sum(stopping_function_output_int, dim=0) - 1
                ) / (len(self.ts) - 1)
                stopping_timestep_vector = []

            s_vector = []
            t_vector = []
            for k, t in enumerate(self.ts):
                s_vector.append(
                    torch.linspace(t, self.T, self.num_steps + 1 - k).to(self.ts.device)
                )
                t_vector.append(
                    t * torch.ones(self.num_steps + 1 - k).to(self.ts.device)
                )
                if use_stopping_time:
                    stopping_timestep_vector.append(
                        stopping_timestep.unsqueeze(0).repeat(self.num_steps + 1 - k, 1)
                    )
            s_vector = torch.cat(s_vector)
            t_vector = torch.cat(t_vector)
            if use_stopping_time:
                stopping_timestep_vector = torch.cat(stopping_timestep_vector, dim=0)
                M_evals_all = self.neural_sde.M(
                    t_vector, s_vector, stopping_timestep_vector
                )
                derivative_M_evals_all = torch.nan_to_num(
                    derivative_M(t_vector, s_vector, stopping_timestep_vector)
                )
                counter = 0
                for k, t in enumerate(self.ts):
                    M_evals[k, k:, :, :, :] = M_evals_all[
                        counter : (counter + self.num_steps + 1 - k), :, :, :
                    ]
                    derivative_M_evals[k, k:, :, :, :] = derivative_M_evals_all[
                        counter : (counter + self.num_steps + 1 - k), :, :, :
                    ]
                    counter += self.num_steps + 1 - k
            else:
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

            if use_stopping_time:
                least_squares_target_integrand_term_1 = torch.einsum(
                    "ijmkl,jml->ijmk",
                    M_evals,
                    self.neural_sde.nabla_f(self.ts, states),
                )[:, :-1, :, :]
            else:
                least_squares_target_integrand_term_1 = torch.einsum(
                    "ijkl,jml->ijmk",
                    M_evals,
                    self.neural_sde.nabla_f(self.ts, states),
                )[:, :-1, :, :]

            if use_stopping_time:
                M_nabla_b_term = (
                    torch.einsum(
                        "ijmkl,jmln->ijmkn",
                        M_evals,
                        self.neural_sde.nabla_b(self.ts, states),
                    )
                    - derivative_M_evals
                )
                least_squares_target_integrand_term_2 = -np.sqrt(
                    self.lmbd
                ) * torch.einsum(
                    "ijmkn,jmn->ijmk",
                    M_nabla_b_term[:, :-1, :, :, :],
                    torch.einsum("ij,abj->abi", sigma_inverse_transpose, noises),
                )
            else:
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

            if use_stopping_time:
                M_evals_final = M_evals[:, -1, :, :, :]
                least_squares_target_terminal = torch.einsum(
                    "imkl,ml->imk",
                    M_evals_final,
                    self.neural_sde.nabla_g(states[-1, :, :]),
                )
            else:
                M_evals_final = M_evals[:, -1, :, :]
                least_squares_target_terminal = torch.einsum(
                    "ikl,ml->imk",
                    M_evals_final,
                    self.neural_sde.nabla_g(states[-1, :, :]),
                )

            if use_stopping_time:
                least_squares_target_integrand_term_1_times_dt = (
                    least_squares_target_integrand_term_1
                    * fractional_timesteps.unsqueeze(0).unsqueeze(3)
                )
                least_squares_target_integrand_term_2_times_sqrt_dt = (
                    least_squares_target_integrand_term_2
                    * torch.sqrt(fractional_timesteps).unsqueeze(0).unsqueeze(3)
                )
                least_squares_target_integrand_term_3_times_dt = (
                    least_squares_target_integrand_term_3
                    * fractional_timesteps.unsqueeze(0).unsqueeze(3)
                )
            else:
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

            if use_stopping_time:
                control_learned = unsqueezed_stop_indicators * self.control(self.ts, states)
                control_target = -unsqueezed_stop_indicators * torch.einsum(
                    "ij,...j->...i",
                    torch.transpose(self.sigma, 0, 1),
                    least_squares_target,
                )
            else:
                control_learned =  self.control(self.ts, states)
                control_target = -torch.einsum(
                    "ij,...j->...i",
                    torch.transpose(self.sigma, 0, 1),
                    least_squares_target,
                )

            if use_stopping_time:
                objective = torch.sum(
                    (control_learned - control_target) ** 2
                    * weight.unsqueeze(0).unsqueeze(2)
                ) / (torch.sum(stop_indicators))
            else:
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

            if use_stopping_time:
                deterministic_integrand_times_dt = (
                    deterministic_integrand * fractional_timesteps
                )
                stochastic_integrand_times_sqrt_dt = stochastic_integrand * torch.sqrt(
                    fractional_timesteps
                )
            else:
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
            if use_stopping_time:
                deterministic_integrand = (
                    deterministic_integrand * stop_indicators[:-1, :]
                )
                stochastic_integrand = stochastic_integrand * stop_indicators[:-1, :]

            if use_stopping_time:
                deterministic_integrand_times_dt = (
                    deterministic_integrand * fractional_timesteps
                )
                stochastic_integrand_times_sqrt_dt = stochastic_integrand * torch.sqrt(
                    fractional_timesteps
                )
            else:
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

            if add_weights:
                weight_2 = weight
            else:
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
            torch.logsumexp(log_weight, dim=0) - torch.log(torch.tensor([log_weight.shape[0]],device=log_weight.device)),
            torch.std(weight),
            stop_indicators,
        )

        
            
            
            


            




        









