import torch

class ConstantBasis():
    """
    Basis obtained by adding constant function to a given basis.
    """
    def __init__(self, basis):
        self.base_basis = basis
        self.dim = self.base_basis.dim
        self.basis_dim = self.base_basis.basis_dim + 1

    def __call__(self, x):
        out = torch.ones([x.size(0),self.basis_dim])
        out[:,1:] = self.base_basis(x)
        return out
    
    def grad(self, x):
        out = torch.zeros([x.size(0),self.basis_dim,self.dim])
        out[:,1:,:] = self.base_basis.grad(x)
        return out
    
    def laplacian(self, x):
        out = torch.zeros([x.size(0),self.basis_dim])
        out[:,1:] = self.base_basis.laplacian(x)
        return out
