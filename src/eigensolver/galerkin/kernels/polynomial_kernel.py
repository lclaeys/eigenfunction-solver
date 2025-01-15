from src.eigensolver.galerkin.kernels.base_kernel import BaseKernel
import torch

class PolynomialKernel(BaseKernel):
    """
    Polynomial kernel, f(x,y) = (constant+(x^Ty)/scale^2)^p
    """
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.scale = params.get('scale',1.0)
        self.constant = params.get('consant', 1.0)
        self.order = params.get('order', 1.0)

    def forward(self, x, y):
        """
            Evaluate kernels with base points x at points y

            Args:
                x (ndarray)[n, d]: first set of points for kernel evaluation
                y (ndarray)[p, d]: second set of points for kernel evaluation
            Returns:
                k_xy (ndarray)[n,p]: k_xy(i,j) = k(x_i,y_j)
        """

        k_xy = (self.constant + x@y.T/self.scale**2)**self.order

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
        xy_term = (self.constant + x@y.T/self.scale**2)**(self.order-1)
        grad_k_xy = self.order/self.scale**2*xy_term[:,:,None]*y[None,:,:]

        return grad_k_xy
    
    def laplacian(self, x, y):
        """
            Evaluate kernel laplacian with base points x at points y

            Args:
                x (ndarray)[n, d]: evaluation points for kernel laplacian evaluation
                y (ndarray)[p, d]: base points for kernel laplacian evaluation
            Returns:
                delta_k_xy (ndarray)[n,p]: delta_k_xy(i,j) = div(grad(k_{y_j}))(x_i)
        """
        y_norm = torch.linalg.norm(y,dim=1)
        delta_k_xy = self.order*(self.order-1)/self.scale**4*(self.constant + x@y.T/self.scale**2)**(self.order-2)*y_norm[None,:]
        return delta_k_xy