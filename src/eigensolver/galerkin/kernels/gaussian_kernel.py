import torch
from src.eigensolver.galerkin.kernels.base_kernel import BaseKernel

# TODO: is this implementation optimal for large number of points and large dimension?
# See: https://github.com/VivienCabannes/laplacian/tree/master/src/klap/kernels

class GaussianKernel(BaseKernel):
    """
    Gaussian kernel, f(x,y) = exp(-||x-y||^2/(2*scale^2))
    """
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.scale = params.get('scale',1.0)

    def forward(self, x, y):
        """
            Evaluate kernels with base points x at points y

            Args:
                x (ndarray)[n, d]: first set of points for kernel evaluation
                y (ndarray)[p, d]: second set of points for kernel evaluation
            Returns:
                k_xy (ndarray)[n,p]: k_xy(i,j) = k(x_i,y_j)
        """
        distances = torch.cdist(x,y)
        k_xy = torch.exp(-distances**2/(2*self.scale**2))

        return k_xy
    
    def grad(self, x, y):
        """
            Evaluate kernel gradient with base points x at points y

            Args:
                x (ndarray)[n, d]: evaluation points for kernel gradient evaluation
                y (ndarray)[p, d]: base points for kernel gradient evaluation
            Returns:
                grad_k_xy (ndarray)[n,p,d]: grad_k_xy(i,j) = grad k_{y_j}(x_i)
        """
        diffs = x[:,None,:]-y[None,:,:]
        k_xy = self.forward(x,y)

        grad_k_xy = -1/(self.scale**2)*diffs*k_xy[:,:,None]

        return grad_k_xy
    
    def laplacian(self, x, y):
        # TODO: what formula is this based on? Is this optimal?
        """
            Evaluate kernel laplacian with base points x at points y

            Args:
                x (ndarray)[n, d]: evaluation points for kernel laplacian evaluation
                y (ndarray)[p, d]: base points for kernel laplacian evaluation
            Returns:
                delta_k_xy (ndarray)[n,p]: delta_k_xy(i,j) = div(grad(k_{y_j}))(x_i)
        """
        distances = torch.cdist(x,y)
        k_xy = torch.exp(-distances**2/(2*self.scale**2))

        delta_k_xy = (-self.dim/self.scale**2 + distances**2/self.scale**4)*k_xy

        return delta_k_xy