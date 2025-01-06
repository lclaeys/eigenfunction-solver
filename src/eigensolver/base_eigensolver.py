class BaseSolver():
    """
    Base class for eigenfunction solvers.

    Initialized using energy object
    """
    def __init__(self, energy, *args, **kwargs):
        self.energy = energy

    def fit(self, data, config):
        """
        Fit the eigenfunctions.

        Args:
            data (ndarray): samples from stationary distribution
        """
        raise NotImplementedError
    
    def predict(self,x):
        """
        Evaluate learned eigenfunction at points x.

        Args:
            x (ndarray)[n,d]: points at which to evaluate
        Returns:
            fx (ndarray)[m,n,d]: learned eigenfunctions evaluated at points x.
        """
        raise NotImplementedError