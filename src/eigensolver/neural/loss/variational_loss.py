import torch
import torch.nn as nn

class VariationalLoss(nn.Module):
    """
    Variational loss used in Zhang et al.
    L(f) = sum_i <f_i, Lf_i>_mu = sum_i E_(x \sim mu)[<grad f_i(x), grad f_i(x)>]
    """

    def __init__(self, beta):
        super(VariationalLoss, self).__init__()
        self.beta = beta

    def forward(self, grad_f):
        """
        Args:
            grad_f (tensor)[N, m, d]: gradient of functions evaluated at N points
        Returns:
            var_loss (tensor): variational loss 
        """
        if self.beta.device != grad_f.device:
            self.beta = self.beta.to(grad_f.device)

        return torch.sum(self.beta * torch.sum(grad_f**2,dim=(0,2))) / grad_f.shape[0]