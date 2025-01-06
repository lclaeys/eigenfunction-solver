class BaseKernel():
    """
    Base kernel class.
    """
    def __init__(self, params, *args, **kwargs):
        self.dim = params.get('dim')

    def forward(self, x, y):
        """
            Evaluate kernels with base points x at points y

            Args:
                x (ndarray)[nx, d]: base points for kernel evaluation
                y (ndarray)[ny, d]: evaluation points for kernel evaluation
            Returns:
                k_xy (ndarray)[nx,ny]: k_xy(i,j) = k(x_i,y_j)
        """
        pass
    
    def grad(self, x, y):
        """
            Evaluate kernel gradient with base points x at points y

            Args:
                x (ndarray)[n, d]: evaluation points for kernel gradient evaluation
                y (ndarray)[p, d]: base points for kernel gradient evaluation
            Returns:
                grad_k_xy (ndarray)[n,p,d]: grad_k_xy(i,j) = grad k_{y_j}(x_i)
        """
        pass
    
    def laplacian(self, x, y):
        """
            Evaluate kernel laplacian with base points x at points y

            Args:
                x (ndarray)[n, d]: evaluation points for kernel laplacian evaluation
                y (ndarray)[p, d]: base points for kernel laplacian evaluation
            Returns:
                delta_k_xy (ndarray)[n,p]: delta_k_xy(i,j) = div(grad(k_{y_j}))(x_i)
        """
        pass

