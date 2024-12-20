class KernelBasis():
    """
    Basis from taking a kernel with some basis points.
    """
    def __init__(self, kernel, basis_points):
        self.kernel = kernel
        self.xp = basis_points
        self.dim = self.kernel.dim
        self.basis_dim = len(basis_points)
    
    def __call__(self, x):
        return self.kernel.forward(x, self.xp)
    
    def grad(self, x):
        return self.kernel.grad(x, self.xp)
    
    def laplacian(self, x):
        return self.kernel.laplacian(x, self.xp)
