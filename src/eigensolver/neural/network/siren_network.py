import torch
import torch.nn as nn

from src.eigensolver.neural.network.orthonormal_layer import OrthonormalLayer

class OrthSIREN(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, inner_prod, momentum = 0.9,omega_0=30.0):
        """
        A simple multi-layer SIREN network.

        Args:
            in_features: Number of input features (e.g., spatial dimensions like 2 or 3).
            hidden_features: Number of hidden units in each layer.
            hidden_layers: Number of hidden layers.
            out_features: Number of output features (e.g., 1 for scalar fields).
            inner_prod (function): inner product for orth layer
            momentum (float): momentum for orth layer
            omega_0: Scaling factor for the first layer.
        """
        super(OrthSIREN, self).__init__()
        self.net = nn.ModuleList()
        
        # First layer with omega_0 scaling
        self.net.append(SIRENLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))
        
        # Hidden layers without omega_0 scaling
        for _ in range(hidden_layers):
            self.net.append(SIRENLayer(hidden_features, hidden_features, is_first=False))
        
        # Output layer without activation
        self.net.append(nn.Linear(hidden_features, out_features))
        nn.init.xavier_uniform_(self.net[-1].weight)  # Optional: Initialize the output layer weights

        self.orth_layer = OrthonormalLayer(inner_prod, momentum)

    def forward(self, x):
        """Forward pass through the network."""
        samples = x.clone().detach()
        samples.requires_grad_(False)
        for layer in self.net[:-1]:  # Apply SIREN layers
            x = layer(x)
        
        x = self.net[-1](x)  # Apply the final linear layer
        fsamples = x.clone().detach()
        fsamples.requires_grad_(False)

        if self.training:
            x = self.orth_layer(x,samples,fsamples)
        else:
            x = self.orth_layer(x)
        
        return x
        
class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        """
        A single SIREN layer with sinusoidal activation.
        
        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            is_first: Whether this is the first layer in the network.
            omega_0: Scaling factor for the first layer.
        """
        super(SIRENLayer, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        # Linear layer
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights according to the SIREN initialization scheme."""
        with torch.no_grad():
            if self.is_first:
                # Scale the weights for the first layer
                self.linear.weight.uniform_(-1 / self.omega_0, 1 / self.omega_0)
            else:
                # Use standard Xavier initialization for subsequent layers
                nn.init.xavier_uniform_(self.linear.weight)
            # Bias can be initialized to zero or a small random value
            self.linear.bias.uniform_(-1e-4, 1e-4)
    
    def forward(self, x):
        """Apply the linear layer followed by the sine activation."""
        return torch.sin(self.omega_0 * self.linear(x)) if self.is_first else torch.sin(self.linear(x))
