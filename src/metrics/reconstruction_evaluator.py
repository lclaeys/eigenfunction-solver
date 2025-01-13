import numpy as np
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
        self.grad = torch.func.grad(func)

    def compute_reconstruction_error(self, x, fx, samples, fsamples, eigvals):
        """
        Compute error in reconstructing func and Lfunc
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
            tensor_x = torch.tensor(x)
            tensor_samples = torch.tensor(samples)

            funcx = np.array(self.batch_func(tensor_x))
            funcsamples = np.array(self.batch_func(tensor_samples))

            grad_funcx = np.array(self.grad(tensor_x))
            laplacian_funcx = np.array(self.laplacian(tensor_x))

        var_funcx = np.mean((funcx - np.mean(funcx))**2)
        
        Lfuncx = -laplacian_funcx + np.sum(grad_funcx*self.energy.grad(x),axis=1) 
        ss_Lfuncx = np.mean(Lfuncx**2)
        
        inner_prods = np.sum(funcsamples[:,None]*fsamples,axis=0)/samples.shape[0]

        errs = []
        L_errs = []

        for i in range(1,fx.shape[1]+1):
            reconstruction = inner_prods[:i]@(fx[:,:i]).T
            errs.append(np.mean((funcx - reconstruction)**2)/var_funcx)

            L_reconstruction = (eigvals[:i]*inner_prods[:i])@(fx[:,:i]).T
            L_errs.append(np.mean((Lfuncx - L_reconstruction)**2)/ss_Lfuncx)

        return np.array(errs), np.array(L_errs)
    
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
            tensor_x = torch.tensor(x)
            tensor_samples = torch.tensor(samples)

            funcx = np.array(self.batch_func(tensor_x))
            funcsamples = np.array(self.batch_func(tensor_samples))

            grad_funcx = np.array(self.grad(tensor_x))
            laplacian_funcx = np.array(self.laplacian(tensor_x))

        var_funcx = np.mean((funcx - np.mean(funcx))**2)
        
        Lfuncx = -laplacian_funcx + np.sum(grad_funcx*self.energy.grad(x),axis=1) 
        ss_Lfuncx = np.mean(Lfuncx**2)
        
        inner_prods = np.sum(funcsamples[:,None]*fsamples,axis=0)/samples.shape[0]

        fig, axes = plt.subplots(fx.shape[1],2,figsize=(10,(fx.shape[1])*3))
       
        for i in range(1,fx.shape[1]+1):
            reconstruction = inner_prods[:i]@(fx[:,:i]).T
            err = np.mean((funcx - reconstruction)**2)/var_funcx

            axes[i-1,0].plot(x, funcx, color='black', label = 'func')
            axes[i-1,0].plot(x, reconstruction, color='blue', ls = '--', label = 'reconstruction')
            axes[i-1,0].set_title(f'Reconstruction of func, err = {err:.3e}, k = {i}')
            axes[i-1,0].legend()

            L_reconstruction = (eigvals[:i]*inner_prods[:i])@(fx[:,:i]).T
            L_err = np.mean((Lfuncx - L_reconstruction)**2)/ss_Lfuncx
            axes[i-1,1].plot(x, Lfuncx, color='black', label = 'Lfunc')
            axes[i-1,1].plot(x, L_reconstruction, color='blue', ls = '--', label = 'reconstruction')
            axes[i-1,1].set_title(f'Reconstruction of Lfunc, err = {L_err:.3e}, k = {i}')
            axes[i-1,1].legend()

        return fig, axes
        


