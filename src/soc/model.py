import torch
import torch.nn as nn

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