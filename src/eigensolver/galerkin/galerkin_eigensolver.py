import numpy as np
import torch
from src.eigensolver.base_eigensolver import BaseSolver
from numpy.linalg import LinAlgError
from torch.utils.data import RandomSampler, DataLoader
from scipy.linalg import eigh

class GalerkinSolver(BaseSolver):
    """
    Galerkin solver from Cabannes et al. 2024
    """
    def __init__(self, energy, samples, params, *args, **kwargs):
        """
        Args:
            energy (BaseEnergy): energy function
            samples (tensor): samples
            params (dict): parameters for solver
                num_samples (int): number of samples to use when estimatating expectations wrt mu
                batch_size (int): number of samples to load at a time when estimating expectations
                verbose (bool): whether to print outputs
        """
        super().__init__(energy, *args, **kwargs)
        self.dim = energy.dim
        self.verbose = params.get('verbose',False)

        # TODO: good choice for samples for energies where sampler is unavailable
        self.num_samples = params.get('num_samples',10000)
        self.batch_size = params.get('batch_size', 10000)

        if self.num_samples // self.batch_size != 0:
            raise AssertionError(f"Number of samples ({self.num_samples}) should be multiple of batch size ({self.batch_size})")

        random_sampler = RandomSampler(samples, num_samples=self.num_samples)
        self.dataloader = DataLoader(samples, batch_size = self.batch_size, sampler = random_sampler)
    
    def fit(self, basis, k = 16, L_reg = 0, phi_reg = 0, seed = 42):
        """
        Fit the eigenfunctions.

        Args:
            basis (Basis): basis functions object
            k (int): number of eigenfunctions to compute
            L_reg (float): regularizer for L
            phi_reg (float): regularizer for phi
            seed (int): random seed
        """
        torch.manual_seed(seed)
        
        L = self.compute_L(basis).double()
        phi0 = self.compute_phi(basis).double()

        error = torch.linalg.eigvalsh(L + L_reg*torch.eye(L.size(0)))[0]
        if error < 0:
            L_reg += -error*10
            if self.verbose:
                print(f'Warning: L not positive definite, adding regularizer {phi_reg:.3e}')

        error = torch.linalg.eigvalsh(phi0 + phi_reg*torch.eye(phi0.size(0)))[0]
        if error < 0:
            phi_reg += -error*1.1
            if self.verbose:
                print(f'Warning: phi not positive definite, adding regularizer {phi_reg:.3e}')

        L = L + L_reg*torch.eye(L.size(0))
        phi = phi0 + phi_reg*torch.eye(phi0.size(0))

        try:
            # Use scipy for generalized eigenvalue problem
            L, phi = L.numpy(), phi.numpy()
            eigvals, eigvecs = eigh(L, phi, subset_by_index=[0, k-1])
            eigvals, eigvecs = torch.tensor(eigvals), torch.tensor(eigvecs)
        
        except LinAlgError as e:
            if self.verbose:
                print('Error solving GEVD')
            return None
    
        # (k,)
        self.eigvals = eigvals

        # (p, k)
        self.eigvecs = eigvecs
        self.basis = basis

        return self
    
    def predict(self,x):
        """
        Evaluate learned eigenfunction at points x.

        Args:
            x (tensor)[n,d]: points at which to evaluate
        Returns:
            fx (tensor)[n,k]: learned eigenfunctions evaluated at points x.
        """
        fx = self.basis(x)@self.eigvecs

        return fx
    
    def predict_grad(self, x):
        """
        Evaluate gradient of learned eigenfunction at points x.

        Args:
            x (tensor)[n,d]: points at which to evaluate
        Returns:
            grad_fx (tensor)[n,k,d]: gradient of learned eigenfunctions evaluated at points x.
        """
        # (n, p, d)
        grad_basis = self.basis.grad(x)

        grad_fx = (grad_basis.transpose(1,2) @ self.eigvecs).transpose(1,2)
        
        return grad_fx
    
    def predict_laplacian(self, x):
        """
        Evaluate laplacian of learned eigenfunction at points x.

        Args:
            x (tensor)[n,d]: points at which to evaluate
        Returns:
            delta_fx (tensor)[n,m]: laplacian of learned eigenfunctions evaluated at points x.
        """
        delta_basis = self.basis.laplacian(x)

        delta_fx = delta_basis @ self.eigvecs

        return delta_fx
    
    def predict_Lf(self, x):
        """
        Evaluate Lf of learned eigenfunction at points x.

        Args:
            x (tensor)[n,d]: points at which to evaluate
        Returns:
            Lfx (tensor)[n,m]: Lf evaluated at points x.
        """

        energy_grad = self.energy.grad(x)

        Lfx = -self.predict_laplacian(x) + torch.bmm(self.predict_grad(x), energy_grad.unsqueeze(2)).squeeze(2)

        return Lfx
    
    def fit_eigvals(self, x):
        """
        fit eigvals using OLS
        Args:
            x (tensor): points to use for fitting
        Returns:
            fitted_eigvals (tensor): k fitted eigvals 
        """
        self.fx = self.predict(x)
        self.Lfx = self.predict_Lf(x)

        self.fitted_eigvals = torch.sum(self.fx*self.Lfx,dim=0)/torch.sum(self.fx**2,dim=0)
        return self.fitted_eigvals

    def compute_L(self, basis):
        """
        Compute the Matrix L given by L_ij = sum_{k=1:n} H(k_xi, k_xj, xk)
        
        Args:
            basis (Basis): basis functions
        Returns:
            L (tensor)[p,p]: matrix L
        """
        
        L = torch.zeros((basis.basis_dim,basis.basis_dim))

        for batch in self.dataloader:
            
            # (batch, p, d)
            grad_basis = basis.grad(batch)

            L += torch.bmm(grad_basis, grad_basis.transpose(1,2)).sum(axis=0)/batch.size(0)
        
        L /= len(self.dataloader)

        return L
    
    def compute_phi(self, basis):
        """
        Compute the Matrix Phi given by Phi_ij = sum_{k=1:n} k_xi(xk) k_xj(xk)
        
        Args:
            basis (Basis): basis functions
        Returns:
            phi (tensor)[p,p]: matrix Phi
        """
        phi = torch.zeros((basis.basis_dim,basis.basis_dim))
        
        for batch in self.dataloader:
            
            # (batch, p)
            batch_basis = basis(batch)

            phi += torch.sum(batch_basis[:,:,None]*batch_basis[:,None,:],dim=0)/batch.size(0)
        
        phi /= len(self.dataloader)

        return phi
        
        

        





