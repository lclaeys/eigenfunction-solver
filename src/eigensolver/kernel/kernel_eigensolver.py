import torch
import importlib
from src.eigensolver.base_eigensolver import BaseSolver

class KernelSolver(BaseSolver):
    """
    Kernel solver from Cabannes et al. 2024
    """
    def __init__(self, energy, sigma, params, *args, **kwargs):
        super().__init__(energy, sigma, *args, **kwargs)
        self.kernel_name = params.get('kernel_name')
        self.kernel_params = params.get('kernel_params')
        self.kernel = importlib.import_module(f"src.eigensolver.kernel.kernels.{self.kernel_name}").create_instance(self.kernel_params)
        self.regularizer = params.get('regularizer',0)
        self.dim = self.kernel.dim

    def fit(self, data, params):
        """
        Fit the eigenfunctions.

        Args:
            data (Tensor)[n,d]: samples from stationary distribution
        """
        self.p = params.get('p')
        x = data
        xp = params.get('xp', x[:self.p])
        self.xp = xp
        
        L = self.compute_L(x,xp)
        phi = self.compute_phi(x,xp)

        psi = torch.linalg.cholesky(phi)
        psi_inv = torch.linalg.inv(psi)
        C = psi_inv @ L @ psi_inv.T

        eigvals, eigvecs = torch.linalg.eigh(C)

        eigvecs = eigvecs.T@psi_inv

        orth_error = torch.mean((eigvecs@phi@eigvecs.T-torch.eye(eigvecs.shape[0]))**2)
        print(f'Orthogonality error: {orth_error}')

        l_error = torch.mean((phi@eigvecs.T@torch.diag(eigvals)@eigvecs@phi.T-L)**2)
        print(f'L error: {l_error}')

        self.eigvals = eigvals
        self.eigvecs = eigvecs
    
    def predict(self,x):
        """
        Evaluate learned eigenfunction at points x.

        Args:
            x (Tensor)[n,d]: points at which to evaluate
        Returns:
            fx (Tensor)[n,m]: learned eigenfunctions evaluated at points x.
        """
        kxy = self.kernel.forward(x,self.xp)

        fx = kxy@self.eigvecs

        return fx
    
    def compute_L(self, x, xp):
        """
        Compute the Matrix L given by L_ij = sum_{k=1:n} H(k_xi, k_xj, xk)
        
        Args:
            x (Tensor)[n,d]: samples from stationary distribution
            xp (Tensor)[p,d]: basis points for kernel 
        Returns:
            L (Tensor)[p,p]: matrix L
        """

        # (n, p)
        delta_k = self.kernel.laplacian(x, xp)

        # (n, p, d)
        grad_k = self.kernel.grad(x,xp)

        # (n, d)
        energy_grad = self.energy.grad(x)

        # (n, p)
        energy_dotprod = torch.bmm(grad_k, energy_grad.unsqueeze(2)).squeeze(2)
        k = self.kernel.forward(x,xp)

        # (n, p, p)
        G = k[:,:,None]*(-self.sigma*delta_k[:,None,:] + energy_dotprod[:,None,:])

        H = 1/2*G + 1/2*G.transpose(1,2)
        
        L = torch.sum(H,dim=0)

        return L
    
    def compute_phi(self, x, xp):
        """
        Compute the Matrix Phi given by Phi_ij = sum_{k=1:n} k_xi(xk) k_xj(xk)
        
        Args:
            x (Tensor)[n,d]: samples from stationary distribution
            xp (Tensor)[p,d]: basis points for kernel 
        Returns:
            phi (Tensor)[p,p]: matrix Phi
        """
        
        # (n, p)
        k = self.kernel.forward(x,xp)

        # (p, p)
        phi = torch.sum(k[:,:,None]*k[:,None,:],dim=0)
        
        return phi + self.regularizer*torch.eye(xp.shape[0])
        
        

        





