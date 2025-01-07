import torch
import torch.nn as nn

class BasicOrthogonalityLoss(nn.Module):
    """
    Basic orthogonality loss:
    \| E_mu [ff^T] - I \|^2
    """

    def __init__(self):
        super(BasicOrthogonalityLoss, self).__init__()

    def forward(self, fx):
        """
        Args:
            fx (torch.tensor)[N, m]: functions evaluated at N points
        Returns:
            loss (torch.tensor): orthogonality loss
        """
        return torch.sum((torch.mean(fx[:,:,None]*fx[:,None,:],dim=0) - torch.eye(fx.shape[1],device=fx.device))**2)

# TODO Check this, add more types of orthogonality loss
class CovOrthogonalityLoss(nn.Module):
    """
    Covariance orthogonality loss:
    \| Cov_mu [f] - I \|^2
    """

    def __init__(self):
        super(CovOrthogonalityLoss, self).__init__()

    def forward(self, fx):
        """
        Args:
            fx (torch.tensor)[N, m]
        """
        return torch.sum((torch.cov(fx) - torch.eye(fx.shape[1],device=fx.device))**2)