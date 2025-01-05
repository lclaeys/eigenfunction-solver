import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    """
    A customizable feedforward neural network.
    """
    def __init__(self, layer_dims):
        """
        Initializes the neural network with the specified layer dimensions.

        Args:
            layer_dims (list): A list of integers where each element specifies the number
                               of units in the corresponding layer of the neural network.
        """
        super(FeedForwardNetwork, self).__init__()
        
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must have at least two layers (input and output).")
        
        # Create a list to hold layers
        layers = []
        for i in range(len(layer_dims) - 1):
            # Add a fully connected layer
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            
            # Add an activation function (ReLU) after every layer except the last
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        
        # Combine the layers into a Sequential module
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output of the network.
        """
        return self.model(x)
    
class ConstantFFN(nn.Module):
    """
    FFN with first output constant
    """
    def __init__(self, layer_dims):
        """
        Initializes the neural network with the specified layer dimensions.

        Args:
            layer_dims (list): A list of integers where each element specifies the number
                               of units in the corresponding layer of the neural network.
        """
        super(ConstantFFN, self).__init__()
        
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must have at least two layers (input and output).")
        
        # Create a list to hold layers
        layers = []

        self.output_dim = layer_dims[-1]
        
        layer_dims[-1] = layer_dims[-1] - 1
        for i in range(len(layer_dims) - 1):
            # Add a fully connected layer
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            
            # Add an activation function (ReLU) after every layer except the last
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        
        # Combine the layers into a Sequential module
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output of the network.
        """
        batch_size = x.size(0)
        output = torch.empty(batch_size, self.output_dim, device=x.device, dtype=x.dtype)
        output[:, 0] = 1.0  # Assign constant value
        output[:, 1:] = self.model(x)  # Fill the remaining dimensions
        return output