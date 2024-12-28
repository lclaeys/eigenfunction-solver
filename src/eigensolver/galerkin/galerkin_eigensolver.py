import numpy as np
import importlib
from src.eigensolver.base_eigensolver import BaseSolver
from numpy.linalg import LinAlgError
from scipy.linalg import eigh

class GalerkinSolver(BaseSolver):
    """
    Galerkin solver from Cabannes et al. 2024
    """
    def __init__(self, energy, sigma, params, *args, **kwargs):
        super().__init__(energy, sigma, *args, **kwargs)
        self.dim = energy.dim
        self.verbose = params.get('verbose',False)

    def fit(self, data, basis, k = 16, L_reg = 0, phi_reg = 0):
        """
        Fit the eigenfunctions.

        Args:
            data (Tensor)[n,d]: samples from stationary distribution
            basis (Basis): basis functions
            k (int): number of eigenfunctions to compute
        """
        x = data
        
        L = self.compute_L(x,basis)
        phi = self.compute_phi(x,basis)

        error = eigh(L, eigvals_only=True, subset_by_index=[0, 0])[0]
        if error < 0:
            L_reg += -error*10
        
        error = eigh(phi, eigvals_only=True, subset_by_index=[0, 0])[0]
        if error < 0:
            phi_reg += -error*1.1
            if self.verbose:
                print(f'Warning: phi not positive definite, adding regularizer {phi_reg:.3e}')

        L = L + L_reg*np.eye(L.shape[0])
        phi = phi + phi_reg*np.eye(phi.shape[0])

        try:
            eigvals, eigvecs = eigh(L, phi, subset_by_index=[0, k])
        except LinAlgError as e:
            if self.verbose:
                print('Error solving GEVD')
            return None
        # try:
        #     psi = np.linalg.cholesky(phi)
        # except RuntimeError:
        #     if self.verbose:
        #             print('Error decomposing phi.')
        #     return None

        # psi_inv = np.linalg.inv(psi)
        # C = psi_inv @ L @ psi_inv.T

        # eigvals, eigvecs = np.linalg.eigh(C)
        # eigvecs = eigvecs.T@psi_inv

        # if self.verbose:
        #     orth_error = np.mean((eigvecs@phi@eigvecs.T-np.eye(eigvecs.shape[0]))**2)
        #     print(f'Orthogonality error: {orth_error}')

        #     l_error = np.mean((phi@eigvecs.T@np.diag(eigvals)@eigvecs@phi.T-L)**2)
        #     print(f'L error: {l_error}')

        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.basis = basis

        return self
    
    def predict(self,x):
        """
        Evaluate learned eigenfunction at points x.

        Args:
            x (Tensor)[n,d]: points at which to evaluate
        Returns:
            fx (Tensor)[n,m]: learned eigenfunctions evaluated at points x.
        """
        fx = self.basis(x)@self.eigvecs

        return fx
    
    def predict_grad(self, x):
        """
        Evaluate gradient of learned eigenfunction at points x.

        Args:
            x (Tensor)[n,d]: points at which to evaluate
        Returns:
            grad_fx (Tensor)[n,m,d]: gradient of learned eigenfunctions evaluated at points x.
        """
        grad_basis = self.basis.grad(x)

        grad_fx = np.transpose((np.transpose(grad_basis,axes=[0,2,1])@self.eigvecs),axes=[0,2,1])
        
        return grad_fx
    
    def predict_laplacian(self, x):
        """
        Evaluate laplacian of learned eigenfunction at points x.

        Args:
            x (Tensor)[n,d]: points at which to evaluate
        Returns:
            delta_fx (Tensor)[n,m]: laplacian of learned eigenfunctions evaluated at points x.
        """
        delta_basis = self.basis.laplacian(x)

        delta_fx = delta_basis@self.eigvecs

        return delta_fx
    
    def predict_Lf(self, x):
        """
        Evaluate Lf of learned eigenfunction at points x.

        Args:
            x (Tensor)[n,d]: points at which to evaluate
        Returns:
            Lfx (Tensor)[n,m]: Lf evaluated at points x.
        """

        energy_grad = self.energy.grad(x)

        Lfx = -self.sigma*self.predict_laplacian(x) + np.matmul(self.predict_grad(x), energy_grad[:,:,None]).squeeze(2)

        return Lfx
    
    def compute_L(self, x, basis):
        """
        Compute the Matrix L given by L_ij = sum_{k=1:n} H(k_xi, k_xj, xk)
        
        Args:
            x (Tensor)[n,d]: samples from stationary distribution
            basis (Basis): basis functions
        Returns:
            L (Tensor)[p,p]: matrix L
        """

        # (n, p)
        delta_basis = basis.laplacian(x)

        # (n, p, d)
        grad_basis = basis.grad(x)

        # (n, d)
        energy_grad = self.energy.grad(x)

        # (n, p)
        energy_dotprod = np.matmul(grad_basis, energy_grad[:,:,None]).squeeze(2)

        """
        # (n, p, p)
        G = basis(x)[:,:,None]*(-self.sigma*delta_basis[:,None,:] + energy_dotprod[:,None,:])

        H = 1/2*G + 1/2*G.transpose(1,2)
        
        L = np.sum(H,axis=0)/x.shape[0]
        """

        L = np.matmul(grad_basis, np.transpose(grad_basis,axes=[0,2,1])).sum(axis=0)/x.shape[0]
        return L
    
    def compute_phi(self, x, basis):
        """
        Compute the Matrix Phi given by Phi_ij = sum_{k=1:n} k_xi(xk) k_xj(xk)
        
        Args:
            x (Tensor)[n,d]: samples from stationary distribution
            basis (Basis): basis functions
        Returns:
            phi (Tensor)[p,p]: matrix Phi
        """

        x_basis = basis(x)

        # (p, p)
        phi = np.sum(x_basis[:,:,None]*x_basis[:,None,:],axis=0)/x.shape[0]
        
        return phi
        
        

        





