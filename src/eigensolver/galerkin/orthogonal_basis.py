class OrthogonalBasis():
    """
    Basis obtained by making previous basis orthogonal to constant function
    """
    def __init__(self, basis, x):
        self.base_basis = basis
        self.dim = self.base_basis.dim
        self.basis_dim = self.base_basis.basis_dim
        self.x = x

    def __call__(self, x):
        out = self.base_basis(x)
        out -= self.base_basis(self.x).mean(axis=0)
        return out
    
    def grad(self, x):
        return self.base_basis.grad(x)
    
    def laplacian(self, x):
        return self.base_basis.laplacian(x)
