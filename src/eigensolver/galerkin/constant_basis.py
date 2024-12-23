import numpy as np

class ConstantBasis():
    """
    Basis obtained by adding constant function to a given basis.
    """
    def __init__(self, basis):
        self.base_basis = basis
        self.dim = self.base_basis.dim
        self.basis_dim = self.base_basis.basis_dim + 1

    def __call__(self, x):
        out = np.ones([x.shape[0],self.basis_dim])
        out[:,1:] = self.base_basis(x)
        return out
    
    def grad(self, x):
        out = np.zeros([x.shape[0],self.basis_dim,self.dim])
        out[:,1:,:] = self.base_basis.grad(x)
        return out
    
    def laplacian(self, x):
        out = np.zeros([x.shape[0],self.basis_dim])
        out[:,1:] = self.base_basis.laplacian(x)
        return out
