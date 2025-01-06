import numpy as np
import importlib
from src.eigensolver.base_eigensolver import BaseSolver
from numpy.linalg import LinAlgError
from scipy.linalg import eigh

class GalerkinSolver(BaseSolver):
    """
    Galerkin solver from Cabannes et al. 2024
    """
    def __init__(self, energy, samples, params, *args, **kwargs):
        """
        Args:
            energy (BaseEnergy): energy function
            samples (ndarray): samples
            params (dict): parameters for solver
                num_samples (int): number of samples to use when estimatating expectations wrt mu
                verbose (bool): whether to print outputs
        """
        super().__init__(energy, *args, **kwargs)
        self.dim = energy.dim
        self.verbose = params.get('verbose',False)
        self.samples = samples
        self.num_samples = params.get('num_samples',10000)
    
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
        self.rng = np.random.default_rng(seed)
        
        L = self.compute_L(basis)
        phi0 = self.compute_phi(basis)

        error = eigh(L, eigvals_only=True, subset_by_index=[0, 0])[0]
        if error < 0:
            L_reg += -error*10
        
        error = eigh(phi0, eigvals_only=True, subset_by_index=[0, 0])[0]
        if error < 0:
            phi_reg += -error*1.1
            if self.verbose:
                print(f'Warning: phi not positive definite, adding regularizer {phi_reg:.3e}')

        L = L + L_reg*np.eye(L.shape[0])
        phi = phi0 + phi_reg*np.eye(phi0.shape[0])

        try:
            eigvals, eigvecs = eigh(L, phi, subset_by_index=[0, k])
        except LinAlgError as e:
            if self.verbose:
                print('Error solving GEVD')
            return None
  
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.basis = basis

        # if self.normalize:
        #     phi = self.compute_phi(basis)
        #     eigvecs_norm = np.diag(self.eigvecs.T@phi@self.eigvecs)
        #     print(eigvecs_norm)
        #     self.eigvecs = self.eigvecs/eigvecs_norm[None,:]

        return self
    
    def predict(self,x):
        """
        Evaluate learned eigenfunction at points x.

        Args:
            x (ndarray)[n,d]: points at which to evaluate
        Returns:
            fx (ndarray)[n,m]: learned eigenfunctions evaluated at points x.
        """
        fx = self.basis(x)@self.eigvecs

        return fx
    
    def predict_grad(self, x):
        """
        Evaluate gradient of learned eigenfunction at points x.

        Args:
            x (ndarray)[n,d]: points at which to evaluate
        Returns:
            grad_fx (ndarray)[n,m,d]: gradient of learned eigenfunctions evaluated at points x.
        """
        grad_basis = self.basis.grad(x)

        grad_fx = np.transpose((np.transpose(grad_basis,axes=[0,2,1])@self.eigvecs),axes=[0,2,1])
        
        return grad_fx
    
    def predict_laplacian(self, x):
        """
        Evaluate laplacian of learned eigenfunction at points x.

        Args:
            x (ndarray)[n,d]: points at which to evaluate
        Returns:
            delta_fx (ndarray)[n,m]: laplacian of learned eigenfunctions evaluated at points x.
        """
        delta_basis = self.basis.laplacian(x)

        delta_fx = delta_basis@self.eigvecs

        return delta_fx
    
    def predict_Lf(self, x):
        """
        Evaluate Lf of learned eigenfunction at points x.

        Args:
            x (ndarray)[n,d]: points at which to evaluate
        Returns:
            Lfx (ndarray)[n,m]: Lf evaluated at points x.
        """

        energy_grad = self.energy.grad(x)

        Lfx = -self.predict_laplacian(x) + np.matmul(self.predict_grad(x), energy_grad[:,:,None]).squeeze(2)

        return Lfx
    
    def fit_eigvals(self, x):
        """
        fit eigvals using OLS
        Args:
            x (ndarray): points to use for fitting
        Returns:
            fitted_eigvals (ndarray): k fitted eigvals 
        """
        self.fx = self.predict(x)
        self.Lfx = self.predict_Lf(x)

        self.fitted_eigvals = np.sum(self.fx*self.Lfx,axis=0)/np.sum(self.fx**2,axis=0)
        return self.fitted_eigvals

    def compute_L(self, basis):
        """
        Compute the Matrix L given by L_ij = sum_{k=1:n} H(k_xi, k_xj, xk)
        
        Args:
            basis (Basis): basis functions
        Returns:
            L (ndarray)[p,p]: matrix L
        """
        x = self.rng.choice(self.samples, size = self.num_samples, axis=0)

        # (n, p)
        #delta_basis = basis.laplacian(x)

        # (n, p, d)
        grad_basis = basis.grad(x)

        # (n, d)
        energy_grad = self.energy.grad(x)

        # (n, p)
        #energy_dotprod = np.matmul(grad_basis, energy_grad[:,:,None]).squeeze(2)

        """
        # (n, p, p)
        G = basis(x)[:,:,None]*(-delta_basis[:,None,:] + energy_dotprod[:,None,:])

        H = 1/2*G + 1/2*G.transpose(1,2)
        
        L = np.sum(H,axis=0)/x.shape[0]
        """

        L = np.matmul(grad_basis, np.transpose(grad_basis,axes=[0,2,1])).sum(axis=0)/x.shape[0]
        return L
    
    def compute_phi(self, basis):
        """
        Compute the Matrix Phi given by Phi_ij = sum_{k=1:n} k_xi(xk) k_xj(xk)
        
        Args:
            basis (Basis): basis functions
        Returns:
            phi (ndarray)[p,p]: matrix Phi
        """
        x = self.rng.choice(self.samples, size = self.num_samples, axis=0)

        x_basis = basis(x)

        # (p, p)
        phi = np.sum(x_basis[:,:,None]*x_basis[:,None,:],axis=0)/x.shape[0]
        
        return phi
        
        

        





