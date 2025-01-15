import torch
import matplotlib.pyplot as plt

class ReconstructionEvaluator():
    """
    Class for evaluating the quality of the learned eigenfunction by seeing how well it can reconstruct a given function
    as well as Lf for that function.
    """
    def __init__(self, energy, func):
        self.energy = energy

        self.batch_func = torch.vmap(func)

        self.hessian = torch.func.hessian(func)
        self.laplacian = torch.vmap(lambda x: torch.diag(self.hessian(x)).sum())

        self.grad = torch.func.jacrev(func)
        self.batch_grad = torch.func.vmap(self.grad)

    def compute_reconstruction_error(self, x, fx, samples, fsamples, eigvals):
        """
        Compute error in reconstructing func and Lfunc
        Args:
            x (torch): points where reconstruction will be evaluated
            fx (torch): eigenfunction estimates at point x
            samples (torch): samples for inner product computation
            fsamples (torch): eigenfunction estimates at samples
            eigvals (torch): eigenvalue estimate
        Returns:
            errs (torch)[k]: MSE of reconstruction / MSE of reconstructing using mean
            L_errs (torch)[k] MSE of reconstruction / SS of Lfunc
        """
        with torch.no_grad():
            funcx = self.batch_func(x)
            funcsamples = self.batch_func(samples)

            grad_funcx = self.batch_grad(x)
            laplacian_funcx = self.laplacian(x)

        var_funcx = torch.mean((funcx - torch.mean(funcx))**2)
        
        Lfuncx = -laplacian_funcx + torch.sum(grad_funcx*self.energy.grad(x),dim=1) 
        ss_Lfuncx = torch.mean(Lfuncx**2)
        
        inner_prods = torch.sum(funcsamples[:,None]*fsamples,dim=0)/samples.size(0)

        errs = []
        L_errs = []

        for i in range(1,fx.shape[1]+1):
            reconstruction = inner_prods[:i]@(fx[:,:i]).T
            errs.append(torch.mean((funcx - reconstruction)**2)/var_funcx)

            L_reconstruction = (eigvals[:i]*inner_prods[:i])@(fx[:,:i]).T
            L_errs.append(torch.mean((Lfuncx - L_reconstruction)**2)/ss_Lfuncx)

        return torch.tensor(errs), torch.tensor(L_errs)
    
    def plot_reconstruction(self,x,fx, samples, fsamples, eigvals):
        """
        Plot reconstruction of func and Lfunc

        Args:
            x (ndarray): points where reconstruction will be evaluated
            fx (ndarray): eigenfunction estimates at point x
            samples (ndarray): samples for inner product computation
            fsamples (ndarray): eigenfunction estimates at samples
            eigvals (ndarray): eigenvalue estimate
        Returns:
            errs (ndarray)[k]: MSE of reconstruction / MSE of reconstructing using mean
            L_errs (ndarray)[k] MSE of reconstruction / SS of Lfunc
        """
        with torch.no_grad():
            funcx = self.batch_func(x)
            funcsamples = self.batch_func(samples)

            grad_funcx = self.batch_grad(x)
            laplacian_funcx = self.laplacian(x)

        var_funcx = torch.mean((funcx - torch.mean(funcx))**2)
        
        Lfuncx = -laplacian_funcx + torch.sum(grad_funcx*self.energy.grad(x),dim=1) 
        ss_Lfuncx = torch.mean(Lfuncx**2)
        
        inner_prods = torch.sum(funcsamples[:,None]*fsamples,dim=0)/samples.size(0)

        fig, axes = plt.subplots(fx.shape[1],2,figsize=(10,(fx.shape[1])*3))
       
        for i in range(1,fx.shape[1]+1):
            reconstruction = inner_prods[:i]@(fx[:,:i]).T
            err = torch.mean((funcx - reconstruction)**2)/var_funcx

            axes[i-1,0].plot(x, funcx, color='black', label = 'func')
            axes[i-1,0].plot(x, reconstruction, color='blue', ls = '--', label = 'reconstruction')
            axes[i-1,0].set_title(f'Reconstruction of func, err = {err:.3e}, k = {i}')
            axes[i-1,0].legend()

            L_reconstruction = (eigvals[:i]*inner_prods[:i])@(fx[:,:i]).T
            L_err = torch.mean((Lfuncx - L_reconstruction)**2)/ss_Lfuncx
            axes[i-1,1].plot(x, Lfuncx, color='black', label = 'Lfunc')
            axes[i-1,1].plot(x, L_reconstruction, color='blue', ls = '--', label = 'reconstruction')
            axes[i-1,1].set_title(f'Reconstruction of Lfunc, err = {L_err:.3e}, k = {i}')
            axes[i-1,1].legend()

        return fig, axes
        


