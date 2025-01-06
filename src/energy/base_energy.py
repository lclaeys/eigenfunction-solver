class BaseEnergy():
    """
    Base class for energy functions
    """

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x):
        """
        Evaluate energy at batch

        Args:
            x (ndarray)[N, d]: points to evaluate at
        Returns:
            energy (ndarray)[N]: energy evaluated at points
        """
        raise NotImplementedError
    
    def grad(self, x):
        """
        Evaluate gradient of energy at batch

        Args:
            x (ndarray)[N, d]: points to evaluate
        Returns:
            grad_x (ndarray)[N, d]: gradient of energy evaluated at points
        """
        raise NotImplementedError
    
    def exact_sample(self, n):
        """
        Compute exact samples from stationary measure

        Args:
            n (ndarray)[shape]: shape of sample
        Returns:
            sample (ndarray)[shape, d]: samples
        """
        raise NotImplementedError