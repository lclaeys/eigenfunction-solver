import torch
import torch.nn as nn
import torch.fft
from src.eigensolver.neural.network.orthonormal_layer import OrthonormalLayer

class FourierLayer(nn.Module):
    """
    A single Fourier layer that:
      1) Applies rFFT on the last dimension.
      2) Multiplies by learned complex weights on the first `modes` frequencies.
      3) Returns an iFFT to go back to physical space.
    """
    def __init__(self, modes, in_channels, out_channels):
        super(FourierLayer, self).__init__()
        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Learned filter: shape (in_channels, out_channels, modes)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        x: (batch_size, in_channels, width)
        returns: (batch_size, out_channels, width)
        """
        batch_size, inC, width = x.shape

        # Real-valued FFT along the last dimension => shape (B, inC, width//2 + 1)
        x_ft = torch.fft.rfft(x, dim=2)
        freq_dim = x_ft.shape[2]

        # Prepare output in Fourier space => (B, outC, freq_dim)
        out_ft = torch.zeros(
            batch_size, self.out_channels, freq_dim,
            dtype=torch.cfloat, device=x.device
        )

        # Limit to the actual number of available frequency modes
        num_modes = min(self.modes, freq_dim)

        # Extract the sub-tensors up to num_modes
        # x_ft_sub: (B, inC, num_modes)
        # weights_sub: (inC, outC, num_modes)
        x_ft_sub = x_ft[:, :, 0:num_modes]
        weights_sub = self.weights[:, :, 0:num_modes]

        # Multiply & sum over input channels => (B, outC, num_modes)
        # Equivalent to: out_ft_sub[b, o, f] = sum_in ( x_ft_sub[b, in, f] * weights_sub[in, o, f] )
        out_ft_sub = torch.einsum('bif, iof->bof', x_ft_sub, weights_sub)

        # Place back into out_ft => (B, outC, freq_dim)
        out_ft[:, :, 0:num_modes] = out_ft_sub

        # Inverse FFT to get back (B, outC, width)
        x_out = torch.fft.irfft(out_ft, n=width, dim=2)
        return x_out

class OrthFNO(nn.Module):
    """
    Minimal 1D FNO:
      - Input shape : (batch_size, input_dim)
      - Output shape: (batch_size, output_dim)
    """
    def __init__(self, modes, in_channels, input_dim, hidden_dim, output_dim, inner_prod, momentum):
        super(OrthFNO, self).__init__()
        self.modes = modes
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Lift to in_channels
        self.fc_in = nn.Linear(input_dim, in_channels)

        # Fourier layers
        self.fourier1 = FourierLayer(modes, in_channels, in_channels)
        self.fourier2 = FourierLayer(modes, in_channels, in_channels)

        # Final MLP
        self.fc_mid = nn.Linear(in_channels * input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.orth_layer = OrthonormalLayer(inner_prod, momentum)
    
    def model_forward(self, x):
        B, D = x.shape  # (batch_size, input_dim)
        # 1) Map to (B, in_channels)
        x_lifted = self.fc_in(x)  # (B, in_channels)

        # 2) Reshape to (B, in_channels, input_dim)
        x_lifted = x_lifted.unsqueeze(-1).expand(B, self.in_channels, self.input_dim)

        # 3) Two Fourier layers (with residuals)
        x_f = self.fourier1(x_lifted) + x_lifted
        x_f = self.fourier2(x_f) + x_f

        # 4) Flatten => (B, in_channels * input_dim)
        x_f = x_f.reshape(B, -1)

        # 5) Nonlinear projection
        x_f = torch.relu(self.fc_mid(x_f))

        # 6) Final linear => (B, output_dim)
        out = self.fc_out(x_f)

        return out
    
    def forward(self, x):
        model_x = self.model_forward(x)
        samples = x.detach().clone()
        funcsamples = self.model_forward(samples)
        out = self.orth_layer(model_x,samples,funcsamples)
        
        return out
    
class FNO(nn.Module):
    """
    Minimal 1D FNO:
      - Input shape : (batch_size, input_dim)
      - Output shape: (batch_size, output_dim)
    """
    def __init__(self, modes, in_channels, input_dim, hidden_dim, output_dim):
        super(FNO, self).__init__()
        self.modes = modes
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Lift to in_channels
        self.fc_in = nn.Linear(input_dim, in_channels)

        # Fourier layers
        self.fourier1 = FourierLayer(modes, in_channels, in_channels)
        self.fourier2 = FourierLayer(modes, in_channels, in_channels)

        # Final MLP
        self.fc_mid = nn.Linear(in_channels * input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Steps:
          1) fc_in => (B, in_channels)
          2) Reshape => (B, in_channels, input_dim)
          3) Apply Fourier layers => (B, in_channels, input_dim)
          4) Flatten => (B, in_channels*input_dim)
          5) fc_mid => (B, hidden_dim)
          6) fc_out => (B, output_dim)
        """
        B, D = x.shape  # (batch_size, input_dim)
        # 1) Map to (B, in_channels)
        x_lifted = self.fc_in(x)  # (B, in_channels)

        # 2) Reshape to (B, in_channels, input_dim)
        x_lifted = x_lifted.unsqueeze(-1).expand(B, self.in_channels, self.input_dim)

        # 3) Two Fourier layers (with residuals)
        x_f = self.fourier1(x_lifted) + x_lifted
        x_f = self.fourier2(x_f) + x_f

        # 4) Flatten => (B, in_channels * input_dim)
        x_f = x_f.reshape(B, -1)

        # 5) Nonlinear projection
        x_f = torch.relu(self.fc_mid(x_f))

        # 6) Final linear => (B, output_dim)
        out = self.fc_out(x_f)
        return out