import torch
import torch.nn as nn

from src.soc.models import FullyConnectedUNet
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
            k=1,
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

        x.requires_grad_(True)

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
                out = torch.stack(outs,dim = 0) / self.norms
                return out, out
        
        model_jac = torch.vmap(torch.func.jacrev(model_fn,has_aux=True))

        t_rev = self.T - t

        regularizer = self.norms[0] * self.inner_prods[0] * reg

        if not self.confining:
            # f = exp(E) f_theta, so grad log f = E + grad log f_theta
            shift = - self.b(None, x) * 2 / self.lmbd
        else:
            shift = 0
        
        if len(x.shape) == 2:
            # (N,k,d) and (N,k)
            Dfx, fx = model_jac(x)

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
            Dfx, fx = model_jac(x_reshaped)

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
        langevin_dt = 0.01
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
        beta = 1.0,
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
        
        if self.neural_sde.joint:
            def model_fn(x):
                out = self.neural_sde.eigf_model(x)
                return out, out
        else:
            def model_fn(x):
                outs = [model(x).squeeze(0) for model in self.neural_sde.eigf_models]
                out = torch.stack(outs,dim = 0)
                return out, out
        
        model_jac = torch.vmap(torch.func.jacrev(model_fn,has_aux=True))

        Dfx, fx = model_jac(self.samples)
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

            return var_loss + 1/beta * orth_loss, var_loss, orth_loss
        
        else:
            var_loss = sq_grad_norms + R_norms
            fx_cov = torch.mean(fx[:,None,:] * fx[:,:,None], dim = 0)
            
            if self.neural_sde.prior == "positive":
                fx_cov[0,0] = torch.logsumexp(fx[:,None,0] + fx[:,0,None],dim=0).exp() / fx.shape[0]

            orth_loss = ((fx_cov - torch.eye(fx.shape[1],device=fx.device))**2).cumsum(dim=0).cumsum(dim=1).diagonal()
            return var_loss + 1/beta * orth_loss, var_loss, orth_loss
    
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
            cutoff_T = 1.0
    ):
        super().__init__()
        self.neural_sde = neural_sde
        self.eigen_sde = eigen_sde
        self.cutoff_T = cutoff_T
        self.T = self.neural_sde.T
        self.lmbd = self.neural_sde.lmbd
        self.lmbd_1 = self.eigen_sde.eigvals[1] -  self.eigen_sde.eigvals[0]
        self.dim = self.neural_sde.dim
        self.device = self.neural_sde.device
        self.confining = self.eigen_sde.confining
        self.b = self.eigen_sde.b
        self.energy = self.eigen_sde.energy
        self.use_learned_control = True
        self.sigma = self.neural_sde.sigma

    def initialize_models(self):
        self.neural_sde.initialize_models()
        self.neural_sde.model = FullyConnectedUNet(
                dim=self.dim+1,
                hdims=self.neural_sde.hdims,
                k = 1
            ).to(self.device)

        self.neural_sde.nabla_V = None
    
        def model_fn(x):
            out = self.eigen_sde.eigf_models[0](x)[0]
            return out, out
        
        self.eigf_jac = torch.vmap(torch.func.jacrev(model_fn,has_aux=True))
        self.epsilon_param = nn.Parameter(torch.tensor(-2.0))
        self.epsilon = torch.log(1 + torch.exp(self.epsilon_param))
    
    def control(self, t, x):
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

        x.requires_grad_(True)

        def model_fn(x):
            out = self.neural_sde.model(x)[0]
            return out, out
        
        model_jac = torch.vmap(torch.func.jacrev(model_fn,has_aux=True))
        
        model_idx = torch.searchsorted(t,self.T - self.cutoff_T) - 1 # find cutoff index
        model_t = t[model_idx:]
        model_x = x[model_idx:]
        model_t_rev = self.T - model_t

        if len(x.shape) == 2:
            x = x.reshape(-1,self.dim)

            model_ts_expand = model_t.reshape(-1, 1).expand(model_x.shape[0], 1)
            model_tx = torch.cat([model_ts_expand, model_x], dim=-1)

            Dfx, fx = model_jac(model_tx)
            Dphix, phix = self.eigf_jac(x)

            # only want spatial derivatives
            Dfx = Dfx[:,1:]

            if not self.confining:
                grad_E = - self.b(None, x) * 2 / self.lmbd
                E = self.energy(x) * 2 / self.lmbd
                Dphix = (Dphix + grad_E) * torch.exp(E).unsqueeze(1)
                phix = phix * torch.exp(E)
            
            v = phix
            v[model_idx:] += torch.exp(-self.lmbd_1 * model_t_rev) * fx

            v = torch.abs(v) + self.epsilon

            grad_v = Dphix
            grad_v[model_idx:] += torch.exp(-self.lmbd_1 * model_t_rev)[:,None] * Dfx

            return - self.lmbd * grad_v / v[:,None]

        if len(x.shape) == 3:
            x_reshaped = x.reshape(-1,self.dim)

            model_ts_repeat = model_t.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], 1)
            model_tx = torch.cat([model_ts_repeat, model_x], dim=-1)
            model_tx_reshape = torch.reshape(model_tx, (-1, model_tx.shape[2]))

            Dfx, fx = model_jac(model_tx_reshape)
            Dphix, phix = self.eigf_jac(x_reshaped)

            # only want spatial derivatives
            Dfx = torch.reshape(Dfx[:,1:], (model_x.shape[0],model_x.shape[1], model_x.shape[2]))
            fx = torch.reshape(fx, (model_x.shape[0],model_x.shape[1]))
            Dphix = torch.reshape(Dphix, (x.shape[0],x.shape[1], x.shape[2]))
            phix = torch.reshape(phix, (x.shape[0],x.shape[1]))

            if not self.confining:
                grad_E = - self.b(None, x) * 2 / self.lmbd
                E = self.energy(x) * 2 / self.lmbd
                Dphix = (Dphix + grad_E) * torch.exp(E).unsqueeze(2)
                phix = phix * torch.exp(E)
            
            v = phix
            v[model_idx:] += torch.exp(-self.lmbd_1 * model_t_rev)[:,None] * fx

            v = torch.abs(v) + self.epsilon

            grad_v = Dphix
            grad_v[model_idx:] += torch.exp(-self.lmbd_1 * model_t_rev)[:,None,None] * Dfx

            return - self.lmbd * grad_v / v[:,:,None]
        
class CombinedSolver(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
            self,
            combined_sde
    ):  
        super().__init__()
        self.neural_sde = combined_sde

    def control(self, t0, x0):
        learned_control = self.neural_sde.control(t0, x0)

        
            
            
            


            




        









