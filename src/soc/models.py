import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
from collections import OrderedDict


class FullyConnectedUNet(torch.nn.Module):
    def __init__(self, dim=2, k=1, hdims=[256, 128, 64], scaling_factor=1.0):
        super().__init__()

        def initialize_weights(layer, scaling_factor):
            for m in layer:
                if isinstance(m, nn.Linear):
                    m.weight.data *= scaling_factor
                    m.bias.data *= scaling_factor

        self.down_0 = nn.Sequential(nn.Linear(dim, hdims[0]), nn.GELU())
        self.down_1 = nn.Sequential(nn.Linear(hdims[0], hdims[1]), nn.GELU())
        self.down_2 = nn.Sequential(nn.Linear(hdims[1], hdims[2]), nn.GELU())
        initialize_weights(self.down_0, scaling_factor)
        initialize_weights(self.down_1, scaling_factor)
        initialize_weights(self.down_2, scaling_factor)

        self.res_0 = nn.Sequential(nn.Linear(dim, k))
        self.res_1 = nn.Sequential(nn.Linear(hdims[0], hdims[0]))
        self.res_2 = nn.Sequential(nn.Linear(hdims[1], hdims[1]))
        initialize_weights(self.res_0, scaling_factor)
        initialize_weights(self.res_1, scaling_factor)
        initialize_weights(self.res_2, scaling_factor)

        self.up_2 = nn.Sequential(nn.Linear(hdims[2], hdims[1]), nn.GELU())
        self.up_1 = nn.Sequential(nn.Linear(hdims[1], hdims[0]), nn.GELU())
        self.up_0 = nn.Sequential(nn.Linear(hdims[0], k), nn.GELU())
        initialize_weights(self.up_0, scaling_factor)
        initialize_weights(self.up_1, scaling_factor)
        initialize_weights(self.up_2, scaling_factor)

    def forward(self, x):
        residual0 = x
        residual1 = self.down_0(x)
        residual2 = self.down_1(residual1)
        residual3 = self.down_2(residual2)

        out2 = self.up_2(residual3) + self.res_2(residual2)
        out1 = self.up_1(out2) + self.res_1(residual1)
        out0 = self.up_0(out1) + self.res_0(residual0)
        return out0
    
class FullyConnectedUNet3(torch.nn.Module):
    def __init__(self, dim=2, k=1, hdims=[256, 128, 64], scaling_factor=1.0):
        super().__init__()

        def initialize_weights(layer, scaling_factor):
            for m in layer:
                if isinstance(m, nn.Linear):
                    m.weight.data *= scaling_factor
                    m.bias.data *= scaling_factor

        self.down_0 = nn.Sequential(nn.Linear(dim, hdims[0]), nn.GELU())
        self.down_1 = nn.Sequential(nn.Linear(hdims[0], hdims[1]), nn.GELU())
        self.down_2 = nn.Sequential(nn.Linear(hdims[1], hdims[2]), nn.GELU())
        initialize_weights(self.down_0, scaling_factor)
        initialize_weights(self.down_1, scaling_factor)
        initialize_weights(self.down_2, scaling_factor)

        self.res_0 = nn.Sequential(nn.Linear(dim, k))
        self.res_1 = nn.Sequential(nn.Linear(hdims[0], hdims[0]))
        self.res_2 = nn.Sequential(nn.Linear(hdims[1], hdims[1]))
        initialize_weights(self.res_0, scaling_factor)
        initialize_weights(self.res_1, scaling_factor)
        initialize_weights(self.res_2, scaling_factor)

        self.up_2 = nn.Sequential(nn.Linear(hdims[2], hdims[1]), nn.GELU())
        self.up_1 = nn.Sequential(nn.Linear(hdims[1], hdims[0]), nn.GELU())
        self.up_0 = nn.Sequential(nn.Linear(hdims[0], k))
        initialize_weights(self.up_0, scaling_factor)
        initialize_weights(self.up_1, scaling_factor)
        initialize_weights(self.up_2, scaling_factor)

    def forward(self, x):
        residual0 = x
        residual1 = self.down_0(x)
        residual2 = self.down_1(residual1)
        residual3 = self.down_2(residual2)

        out2 = self.up_2(residual3) + self.res_2(residual2)
        out1 = self.up_1(out2) + self.res_1(residual1)
        out0 = self.up_0(out1) + self.res_0(residual0)
        return out0
    
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class RingFullyConnectedUNet(nn.Module):
    def __init__(self, dim=2, k=1, hdims=[256, 128, 64], scaling_factor=1.0):
        super().__init__()

        def initialize_weights(layer, scaling_factor):
            for m in layer:
                if isinstance(m, nn.Linear):
                    m.weight.data *= scaling_factor
                    m.bias.data *= scaling_factor

        input_dim = 3  # radius + cos(theta) + sin(theta)

        self.down_0 = nn.Sequential(
            nn.Linear(input_dim, hdims[0]),
            nn.GELU()
        )
        self.down_1 = nn.Sequential(
            nn.Linear(hdims[0], hdims[1]),
            nn.GELU()
        )
        self.down_2 = nn.Sequential(
            nn.Linear(hdims[1], hdims[2]),
            nn.GELU()
        )

        self.res_0 = nn.Sequential(nn.Linear(input_dim, k))
        self.res_1 = nn.Sequential(nn.Linear(hdims[0], hdims[0]))
        self.res_2 = nn.Sequential(nn.Linear(hdims[1], hdims[1]))

        self.up_2 = nn.Sequential(
            nn.Linear(hdims[2], hdims[1]),
            nn.GELU()
        )
        self.up_1 = nn.Sequential(
            nn.Linear(hdims[1], hdims[0]),
            nn.GELU()
        )
        self.up_0 = nn.Sequential(
            nn.Linear(hdims[0], k)
        )

        # Initialize all weights
        for block in [self.down_0, self.down_1, self.down_2,
                      self.res_0, self.res_1, self.res_2,
                      self.up_0, self.up_1, self.up_2]:
            initialize_weights(block, scaling_factor)

    def forward(self, x):
        # x: [..., 2] in Cartesian coordinates
        radius = x.norm(dim=-1, keepdim=True)  # [..., 1]
        angle = torch.atan2(x[..., 1], x[..., 0])  # [...]
        cos_theta = torch.cos(angle).unsqueeze(-1)
        sin_theta = torch.sin(angle).unsqueeze(-1)

        # New input representation: [r, cos(θ), sin(θ)]
        x_reparam = torch.cat([radius, cos_theta, sin_theta], dim=-1)

        # Forward U-Net
        residual0 = x_reparam
        residual1 = self.down_0(x_reparam)
        residual2 = self.down_1(residual1)
        residual3 = self.down_2(residual2)

        out2 = self.up_2(residual3) + self.res_2(residual2)
        out1 = self.up_1(out2) + self.res_1(residual1)
        out0 = self.up_0(out1) + self.res_0(residual0)

        return out0

    

class FullyConnectedUNet2(torch.nn.Module):
    def __init__(self, dim=2, hdims=[256, 128, 64], scaling_factor=1.0):
        super().__init__()

        def initialize_weights(layer, scaling_factor):
            for m in layer:
                if isinstance(m, nn.Linear):
                    m.weight.data *= scaling_factor
                    m.bias.data *= scaling_factor

        self.down_0 = nn.Sequential(nn.Linear(dim + 2, hdims[0]), nn.ReLU())
        self.down_1 = nn.Sequential(nn.Linear(hdims[0], hdims[1]), nn.ReLU())
        self.down_2 = nn.Sequential(nn.Linear(hdims[1], hdims[2]), nn.ReLU())
        initialize_weights(self.down_0, scaling_factor)
        initialize_weights(self.down_1, scaling_factor)
        initialize_weights(self.down_2, scaling_factor)

        self.res_0 = nn.Sequential(nn.Linear(dim + 2, dim))
        self.res_1 = nn.Sequential(nn.Linear(hdims[0], hdims[0]))
        self.res_2 = nn.Sequential(nn.Linear(hdims[1], hdims[1]))
        initialize_weights(self.res_0, scaling_factor)
        initialize_weights(self.res_1, scaling_factor)
        initialize_weights(self.res_2, scaling_factor)

        self.up_2 = nn.Sequential(nn.Linear(hdims[2], hdims[1]), nn.ReLU())
        self.up_1 = nn.Sequential(nn.Linear(hdims[1], hdims[0]), nn.ReLU())
        self.up_0 = nn.Sequential(nn.Linear(hdims[0], dim))
        initialize_weights(self.up_0, scaling_factor)
        initialize_weights(self.up_1, scaling_factor)
        initialize_weights(self.up_2, scaling_factor)

    def forward(self, x):
        residual0 = x
        residual1 = self.down_0(x)
        residual2 = self.down_1(residual1)
        residual3 = self.down_2(residual2)

        out2 = self.up_2(residual3) + self.res_2(residual2)
        out1 = self.up_1(out2) + self.res_1(residual1)
        out0 = self.up_0(out1) + self.res_0(residual0)
        return out0


class SpectralUNet(nn.Module):
    def __init__(self, dim=2, k=1, hdims=[256, 128, 64], scaling_factor=1.0):
        super().__init__()

        def initialize_weights(layer, scaling_factor):
            for m in layer:
                if isinstance(m, nn.Linear):
                    m.weight.data *= scaling_factor
                    m.bias.data *= scaling_factor

        self.down_0 = nn.Sequential(
            spectral_norm(nn.Linear(dim, hdims[0])),
            nn.Softplus()
        )
        self.down_1 = nn.Sequential(
            spectral_norm(nn.Linear(hdims[0], hdims[1])),
            nn.Softplus()
        )
        self.down_2 = nn.Sequential(
            spectral_norm(nn.Linear(hdims[1], hdims[2])),
            nn.Softplus()
        )
        initialize_weights(self.down_0, scaling_factor)
        initialize_weights(self.down_1, scaling_factor)
        initialize_weights(self.down_2, scaling_factor)

        self.res_0 = nn.Sequential(spectral_norm(nn.Linear(dim, k)))
        self.res_1 = nn.Sequential(spectral_norm(nn.Linear(hdims[0], hdims[0])))
        self.res_2 = nn.Sequential(spectral_norm(nn.Linear(hdims[1], hdims[1])))
        initialize_weights(self.res_0, scaling_factor)
        initialize_weights(self.res_1, scaling_factor)
        initialize_weights(self.res_2, scaling_factor)

        self.up_2 = nn.Sequential(
            spectral_norm(nn.Linear(hdims[2], hdims[1])),
            nn.Softplus()
        )
        self.up_1 = nn.Sequential(
            spectral_norm(nn.Linear(hdims[1], hdims[0])),
            nn.Softplus()
        )
        self.up_0 = nn.Sequential(
            spectral_norm(nn.Linear(hdims[0], k))  # No Softplus here to allow unconstrained output
        )
        initialize_weights(self.up_0, scaling_factor)
        initialize_weights(self.up_1, scaling_factor)
        initialize_weights(self.up_2, scaling_factor)

    def forward(self, x):
        residual0 = x
        residual1 = self.down_0(x)
        residual2 = self.down_1(residual1)
        residual3 = self.down_2(residual2)

        out2 = self.up_2(residual3) + self.res_2(residual2)
        out1 = self.up_1(out2) + self.res_1(residual1)
        out0 = self.up_0(out1) + self.res_0(residual0)
        return out0

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class SIREN(nn.Module):
    def __init__(self, dim=2, k=1, hdims=[128,128,128]):
        super().__init__()

        def sine_init(m):
            with torch.no_grad():
                if hasattr(m, 'weight'):
                    num_input = m.weight.size(-1)
                    # See supplement Sec. 1.5 for discussion of factor 30
                    m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


        def first_layer_sine_init(m):
            with torch.no_grad():
                if hasattr(m, 'weight'):
                    num_input = m.weight.size(-1)
                    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
                    m.weight.uniform_(-1 / num_input, 1 / num_input)
        
        self.net = []

        self.net.append(nn.Sequential(nn.Linear(dim,hdims[0]),Sine()))
        for i in range(len(hdims)-1):
            self.net.append(nn.Sequential(
                nn.Linear(hdims[i], hdims[i+1]), Sine()
            ))
        
        self.net.append(nn.Sequential(nn.Linear(hdims[-1], k)))

        self.net = nn.Sequential(*self.net)

        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)

        self.shift = nn.Parameter(torch.zeros(dim))
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Automatically expand shift and scale to match x's dimensions
        shape = [1] * (x.dim() - 1) + [-1]  # e.g. [1, 1, ..., dim]
        shift = self.shift.view(*shape)
        scale = self.scale.view(*shape)

        out = self.net((x-shift)/scale)
        return out
    
# different activation functions
class GaussianActivation(nn.Module):
    def __init__(self, a=1.):
        super().__init__()
        self.a = a

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))
    
class GaussianNet(nn.Module):
    def __init__(self, dim=2, k=1, hdims=[128,128,128]):
        super().__init__()
        
        self.net = []

        self.net.append(nn.Sequential(nn.Linear(dim,hdims[0]),GaussianActivation()))
        for i in range(len(hdims)-1):
            self.net.append(nn.Sequential(
                nn.Linear(hdims[i], hdims[i+1]), GaussianActivation()
            ))
        
        self.net.append(nn.Sequential(nn.Linear(hdims[-1], k)))

        self.net = nn.Sequential(*self.net)

        self.shift = nn.Parameter(torch.zeros(dim))
        self.scale = nn.Parameter(torch.ones(dim))
        # Apply custom initialization
        self._initialize_weights(dim)

    def _initialize_weights(self, dim):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-1/dim, b=1/dim)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # Automatically expand shift and scale to match x's dimensions
        shape = [1] * (x.dim() - 1) + [-1]  # e.g. [1, 1, ..., dim]
        shift = self.shift.view(*shape)
        scale = self.scale.view(*shape)

        out = self.net((x-shift)/scale)
        return out