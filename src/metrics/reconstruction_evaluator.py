import numpy as np
import torch

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

    def compute_reconstruction_error(self, x, fx, samples, fsamples):
        """
        
        Compute error in reconstructing func
        Args:
            x (ndarray): points where reconstruction will be evaluated
            fx (ndarray): eigenfunction estimates at point x
            samples (ndarray): samples for inner product computation
            fsamples (ndarray): eigenfunction estimates at samples
        Returns:
            errs (ndarray)[k]: MSE of reconstruction / MSE of reconstructing using mean
        """
        with torch.no_grad():
            tensor_x = torch.tensor(x)
            tensor_samples = torch.tensor(samples)

            funcx = np.array(self.batch_func(tensor_x))
            funcsamples = np.array(self.batch_func(tensor_samples))

        var_funcx = np.mean((funcx - np.mean(funcx))**2)
        inner_prods = np.sum(funcsamples[:,None]*fsamples,axis=0)/x.shape[0]

        errs = []

        for i in range(1,fx.shape[1]):
            reconstruction = inner_prods[:i]@(fx[:,:i]).T
            errs.append(np.mean((funcx - reconstruction)**2)/var_funcx)

        return np.array(errs)

    def compute_L_reconstruction_error(self, x, fx, samples, fsamples, eigvals):
        """
        
        Compute error in reconstructing L func
        Args:
            x (ndarray): points where reconstruction will be evaluated
            fx (ndarray): eigenfunction estimates at point x
            samples (ndarray): samples for inner product computation
            fsamples (ndarray): eigenfunction estimates at samples
            eigvals (ndarray): eigenvalue estimates
        Returns:
            errs (ndarray)[k]: MSE of reconstruction / SS of Lfx (assuming mean 0)
        """
        with torch.no_grad():
            tensor_x = torch.tensor(x)
            tensor_samples = torch.tensor(samples)

            funcsamples = np.array(self.batch_func(tensor_samples))
            grad_funcx = np.array(self.grad(tensor_x))
            laplacian_funcx = np.array(self.laplacian(tensor_x))

        Lfuncx = -laplacian_funcx + np.sum(grad_funcx*self.energy.grad(x),axis=1) 
        
        ss_funcx = np.mean(Lfuncx**2)

        inner_prods = np.sum(funcsamples[:,None]*fsamples,axis=0)/x.shape[0]

        errs = []

        for i in range(1,fx.shape[1]):
            reconstruction = (eigvals[:i]*inner_prods[:i])@(fx[:,:i]).T
            errs.append(np.mean((Lfuncx - reconstruction)**2)/ss_funcx)

        return np.array(errs)
        
        


