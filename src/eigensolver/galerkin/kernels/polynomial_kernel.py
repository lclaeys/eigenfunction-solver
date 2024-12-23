import numpy as np
from src.eigensolver.galerkin.kernels.base_kernel import BaseKernel

class PolynomialKernel(BaseKernel):
    """
    Polynomial kernel, f(x,y) = (beta+(x^Ty)/alpha^2)^p
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
                x (Tensor)[n, d]: first set of points for kernel evaluation
                y (Tensor)[p, d]: second set of points for kernel evaluation
            Returns:
                k_xy (Tensor)[n,p]: k_xy(i,j) = k(x_i,y_j)
        """

        k_xy = (self.constant + x@y.T/self.scale**2)**self.order

        return k_xy
    
    def grad(self, x, y):
        """
            Evaluate kernel gradient with base points x at points y

            Args:
                x (Tensor)[n, d]: evaluation points for kernel gradient evaluation
                y (Tensor)[p, d]: base points for kernel gradient evaluation
            Returns:
                grad_k_xy (Tensor)[n,p,d]: grad_k_xy(i,j) = grad k_{y_j}(x_i)
        """
        xy_term = (self.constant + x@y.T/self.scale**2)**(self.order-1)
        grad_k_xy = self.order/self.scale**2*xy_term[:,:,None]*y[None,:,:]

        return grad_k_xy
    
    def laplacian(self, x, y):
        """
            Evaluate kernel laplacian with base points x at points y

            Args:
                x (Tensor)[n, d]: evaluation points for kernel laplacian evaluation
                y (Tensor)[p, d]: base points for kernel laplacian evaluation
            Returns:
                delta_k_xy (Tensor)[n,p]: delta_k_xy(i,j) = div(grad(k_{y_j}))(x_i)
        """
        y_norm = np.linalg.norm(y,axis=1)
        delta_k_xy = self.order*(self.order-1)/self.scale**4*(self.constant + x@y.T/self.scale**2)**(self.order-2)*y_norm[None,:]
        return delta_k_xy

def create_instance(params):
    kernel = PolynomialKernel(params)
    return kernel