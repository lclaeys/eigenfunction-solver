class BaseSolver():
    """
    Base class for eigenfunction solvers.

    Initialed using energy object and sigma > 0.
    """
    def __init__(self, energy, sigma, *args, **kwargs):
        self.energy = energy
        self.sigma = sigma

    def fit(self, data, config):
        """
        Fit the eigenfunctions.

        Args:
            data (Tensor): samples from stationary distribution
        """
        raise NotImplementedError
    
    def predict(self,x):
        """
        Evaluate learned eigenfunction at points x.

        Args:
            x (Tensor)[n,d]: points at which to evaluate
        Returns:
            fx (Tensor)[m,n,d]: learned eigenfunctions evaluated at points x.
        """
        raise NotImplementedError