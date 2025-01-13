import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    """
    Simple feed forward neural network with customizable layer dimensions.
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
            
            # Add an activation function (Tanh) after every layer except the last
            if i < len(layer_dims) - 2:
                layers.append(nn.Tanh())
        
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
    FFN with first output constant equal to 1 (first eigenfunction is constant)
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
            
            # Add an activation function (Tanh) after every layer except the last
            if i < len(layer_dims) - 2:
                layers.append(nn.Tanh())
        
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
        
        if len(x.shape) == 1: # unbatched input
            # Generate the constant column
            constant_column = torch.ones((1,), device=x.device, dtype=x.dtype)  # Shape: (batch_size, 1)
            
            # Get the remaining dimensions from the model
            model_output = self.model(x)  # Shape: (batch_size, output_dim - 1)
            
            # Concatenate
            output = torch.cat((constant_column, model_output), dim=0)  # Shape: (batch_size, output_dim)
            
            return output

        if len(x.shape) == 2: # batched input
            # Generate the constant column
            constant_column = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)  # Shape: (batch_size, 1)
            
            # Get the remaining dimensions from the model
            model_output = self.model(x)  # Shape: (batch_size, output_dim - 1)
            
            # Concatenate along the second dimension
            output = torch.cat((constant_column, model_output), dim=1)  # Shape: (batch_size, output_dim)
            
            return output

    
class ConstantLinearFFN(nn.Module):
    """
    FFN with first output constant equal to 1 (first eigenfunction is constant) and second output equal to x (for testing)
    """
    def __init__(self, layer_dims):
        """
        Initializes the neural network with the specified layer dimensions.

        Args:
            layer_dims (list): A list of integers where each element specifies the number
                               of units in the corresponding layer of the neural network.
        """
        super(ConstantLinearFFN, self).__init__()
        
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must have at least two layers (input and output).")
        
        # Create a list to hold layers
        layers = []

        self.output_dim = layer_dims[-1]
        
        layer_dims[-1] = layer_dims[-1] - 2
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
        output[:, 1] = x[:,0]
        output[:, 2:] = self.model(x)  # Fill the remaining dimensions
        return output