import torch

class KernelBasis():
    """
    Kernel Basis
    """
    def __init__(self, kernel, basis_points):
        self.kernel = kernel
        self.xp = basis_points
        self.dim = self.kernel.dim
        self.basis_dim = len(basis_points)
    
    def __call__(self, x):
        if x.dtype != self.xp.dtype:
            self.xp = self.xp.to(x.dtype)

        return self.kernel.forward(x, self.xp)
    
    def grad(self, x):
        if x.dtype != self.xp.dtype:
            self.xp = self.xp.to(x.dtype)
            
        return self.kernel.grad(x, self.xp)
    
    def laplacian(self, x):
        if x.dtype != self.xp.dtype:
            self.xp = self.xp.to(x.dtype)
            
        return self.kernel.laplacian(x, self.xp)
