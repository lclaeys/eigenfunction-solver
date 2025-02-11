import torch
import torch.nn as nn

class OrthonormalLayer(nn.Module):
    """
    Layer which has the goal of making the output orthonormal w.r.t. some inner product.

    It is essential that during training, this 
    """
    def __init__(self, inner_prod, momentum):
        super(OrthonormalLayer, self).__init__()
        self.inner_prod = inner_prod  # Function to compute the inner product
        self.running_L_inv = None
        self.momentum = momentum

    def forward(self, x, samples=None, funcsamples = None):
        if self.training:
            G = self.inner_prod(samples, funcsamples, funcsamples)

            # Cholesky decomposition of the Gram matrix
            L = torch.linalg.cholesky(G)

            # Transform the raw outputs to make them orthonormal
            L_inv = torch.linalg.inv(L)  # Compute the inverse of L
            orthonormal_output = x @ L_inv.T  # Transform the outputs

            if self.running_L_inv is None:
                self.running_L_inv = L_inv
            else:
                self.running_L_inv = self.momentum * self.running_L_inv + (1-self.momentum) * L_inv

            return orthonormal_output
        else:
            orthonormal_output = x @ self.running_L_inv.T
            
            return orthonormal_output
