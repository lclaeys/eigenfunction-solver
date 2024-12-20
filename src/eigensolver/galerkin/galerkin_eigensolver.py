import torch
import importlib
from src.eigensolver.base_eigensolver import BaseSolver
from numpy.linalg import LinAlgError

class GalerkinSolver(BaseSolver):
    """
    Galerkin solver from Cabannes et al. 2024
    """
    def __init__(self, energy, sigma, params, *args, **kwargs):
        super().__init__(energy, sigma, *args, **kwargs)
        self.regularizer = params.get('regularizer',0)
        self.dim = energy.dim

    def fit(self, data, basis, params):
        """
        Fit the eigenfunctions.

        Args:
            data (Tensor)[n,d]: samples from stationary distribution
            basis (Basis): basis functions
        """
        x = data
        
        L = self.compute_L(x,basis)
        phi = self.compute_phi(x,basis)

        try:
            psi = torch.linalg.cholesky(phi)
        except RuntimeError:
            #print('Fitting error: the matrix phi is not positive definite.')
            return None
        
        psi_inv = torch.linalg.inv(psi)
        C = psi_inv @ L @ psi_inv.T

        eigvals, eigvecs = torch.linalg.eigh(C)

        eigvecs = eigvecs.T@psi_inv

        orth_error = torch.mean((eigvecs@phi@eigvecs.T-torch.eye(eigvecs.shape[0]))**2)
        #print(f'Orthogonality error: {orth_error}')

        l_error = torch.mean((phi@eigvecs.T@torch.diag(eigvals)@eigvecs@phi.T-L)**2)
        #print(f'L error: {l_error}')

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
        fx = self.basis(x)@self.eigvecs.T

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

        grad_fx = (grad_basis.transpose(1,2)@self.eigvecs.T).transpose(1,2)
        
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

        delta_fx = delta_basis@self.eigvecs.T

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

        Lfx = -self.sigma*self.predict_laplacian(x) + torch.bmm(self.predict_grad(x), energy_grad.unsqueeze(2)).squeeze(2)

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
        energy_dotprod = torch.bmm(grad_basis, energy_grad.unsqueeze(2)).squeeze(2)

        # (n, p, p)
        G = basis(x)[:,:,None]*(-self.sigma*delta_basis[:,None,:] + energy_dotprod[:,None,:])

        H = 1/2*G + 1/2*G.transpose(1,2)
        
        L = torch.sum(H,dim=0)/x.shape[0]

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
        phi = torch.sum(x_basis[:,:,None]*x_basis[:,None,:],dim=0)/x.shape[0]
        
        return phi + self.regularizer*torch.eye(basis.basis_dim)
        
        

        





