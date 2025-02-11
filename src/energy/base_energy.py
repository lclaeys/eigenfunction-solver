import torch

class BaseEnergy():
    """
    Base class for energy functions
    """

    def __init__(self, *args, **kwargs):
        self.non_confining = False

    def forward(self, x):
        """
        Evaluate energy at batch
        If distribution is \mu \prop exp(-E(x)), then this function should return E(x)

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
    
    def inner_prod(self, samples, fsamples, gsamples):
        """
        Compute inner product based on samples form the stationary measure.
        If E is confining and admits a stationary density, we define
        <f,g> = E[f(x)g(x)]         where x ~ mu(x) = exp(-E(x))

        If E is nonconfining, but -E is, then we instead let
        <f,g> = E[f(x)g(x)exp(-2E)] where x ~ mu(x) = exp(E(x))
        In each case, this yields an inner product for which L (or K) is self-adjoint.

        Args:
            samples (tensor)[N,d]: samples from mu
            fsamples (tensor)[N,n1] or [N,n1,d]: first functions evaluated at samples
            gsamples (tensor)[N,n2] or [N,n2,d]: second functions evaluated at samples
        Returns:
            inner_prods (tensor)[n1,n2]: inner products between first and second functions based on samples
        """
        assert fsamples.shape[0] == gsamples.shape[0] and fsamples.dim() == gsamples.dim()
        
        if not self.non_confining:
            if fsamples.dim() == 2:
                return (fsamples[:,:,None] * gsamples[:,None,:]).mean(dim=0)
            elif fsamples.dim() == 3:
                return torch.bmm(fsamples, gsamples.transpose(1,2)).mean(dim=0)
        
        else:
            # (N,)
            weight = torch.exp(-2*self.forward(samples))

            if fsamples.dim() == 2:
                return (fsamples[:,:,None] * gsamples[:,None,:] * weight[:,None,None]).mean(dim=0)
            elif fsamples.dim() == 3:
                return (weight[:,None,None] * torch.bmm(fsamples, gsamples.transpose(1,2))).mean(dim=0)