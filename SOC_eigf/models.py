"""

Code for the different network architectures. UNet, SigmoidMLP and control classes copied from https://github.com/facebookresearch/SOC-matching.

"""

import torch
import torch.nn as nn

class FullyConnectedUNet(torch.nn.Module):
    def __init__(self, dim_in=2, dim_out=2, hdims=[256, 128, 64], scaling_factor=1.0, reg=0.0):
        super().__init__()

        def initialize_weights(layer, scaling_factor):
            for m in layer:
                if isinstance(m, nn.Linear):
                    m.weight.data *= scaling_factor
                    m.bias.data *= scaling_factor

        self.down_0 = nn.Sequential(nn.Linear(dim_in, hdims[0]), nn.ReLU())
        self.down_1 = nn.Sequential(nn.Linear(hdims[0], hdims[1]), nn.ReLU())
        self.down_2 = nn.Sequential(nn.Linear(hdims[1], hdims[2]), nn.ReLU())
        initialize_weights(self.down_0, scaling_factor)
        initialize_weights(self.down_1, scaling_factor)
        initialize_weights(self.down_2, scaling_factor)

        self.res_0 = nn.Sequential(nn.Linear(dim_in, dim_out))
        self.res_1 = nn.Sequential(nn.Linear(hdims[0], hdims[0]))
        self.res_2 = nn.Sequential(nn.Linear(hdims[1], hdims[1]))
        initialize_weights(self.res_0, scaling_factor)
        initialize_weights(self.res_1, scaling_factor)
        initialize_weights(self.res_2, scaling_factor)

        self.up_2 = nn.Sequential(nn.Linear(hdims[2], hdims[1]), nn.ReLU())
        self.up_1 = nn.Sequential(nn.Linear(hdims[1], hdims[0]), nn.ReLU())
        self.up_0 = nn.Sequential(nn.Linear(hdims[0], dim_out), nn.ReLU())
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
    
class FullyConnectedUNet2(torch.nn.Module):
    def __init__(self, dim_in=2, dim_out=2, hdims=[256, 128, 64], scaling_factor=1.0, reg=0.0):
        super().__init__()

        def initialize_weights(layer, scaling_factor):
            for m in layer:
                if isinstance(m, nn.Linear):
                    m.weight.data *= scaling_factor
                    m.bias.data *= scaling_factor

        self.down_0 = nn.Sequential(nn.Linear(dim_in, hdims[0]), nn.ReLU())
        self.down_1 = nn.Sequential(nn.Linear(hdims[0], hdims[1]), nn.ReLU())
        self.down_2 = nn.Sequential(nn.Linear(hdims[1], hdims[2]), nn.ReLU())
        initialize_weights(self.down_0, scaling_factor)
        initialize_weights(self.down_1, scaling_factor)
        initialize_weights(self.down_2, scaling_factor)

        self.res_0 = nn.Sequential(nn.Linear(dim_in, dim_out))
        self.res_1 = nn.Sequential(nn.Linear(hdims[0], hdims[0]))
        self.res_2 = nn.Sequential(nn.Linear(hdims[1], hdims[1]))
        initialize_weights(self.res_0, scaling_factor)
        initialize_weights(self.res_1, scaling_factor)
        initialize_weights(self.res_2, scaling_factor)

        self.up_2 = nn.Sequential(nn.Linear(hdims[2], hdims[1]), nn.ReLU())
        self.up_1 = nn.Sequential(nn.Linear(hdims[1], hdims[0]), nn.ReLU())
        self.up_0 = nn.Sequential(nn.Linear(hdims[0], dim_out))
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
    
class SigmoidMLP(torch.nn.Module):
    def __init__(self, dim=10, hdims=[128, 128], gamma=3.0, scaling_factor=1.0, reg=0.0):
        super().__init__()

        self.dim = dim
        self.gamma = gamma
        self.sigmoid_layers = nn.Sequential(
            nn.Linear(2, hdims[0]),
            nn.ReLU(),
            nn.Linear(hdims[0], hdims[1]),
            nn.ReLU(),
            nn.Linear(hdims[1], dim**2),
        )

        self.scaling_factor = scaling_factor
        for m in self.sigmoid_layers:
            if isinstance(m, nn.Linear):
                m.weight.data *= self.scaling_factor
                m.bias.data *= self.scaling_factor

    def forward(self, t, s):
        ts = torch.cat((t.unsqueeze(1), s.unsqueeze(1)), dim=1)
        sigmoid_layers_output = self.sigmoid_layers(ts).reshape(-1, self.dim, self.dim)
        exp_factor = (
            torch.exp(self.gamma * (ts[:, 1] - ts[:, 0])).unsqueeze(1).unsqueeze(2)
        )
        identity = torch.eye(self.dim).unsqueeze(0).to(ts.device)
        output = (1 / exp_factor) * identity.repeat(ts.shape[0], 1, 1) + (
            1 - 1 / exp_factor
        ) * sigmoid_layers_output
        return output
    
class GELUNET(torch.nn.Module):
    def __init__(self, dim=2, k=1, hdims=[256, 128, 64], scaling_factor=1.0, reg=0.0):
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

class LinearControl:
    def __init__(self, u, T):
        self.u = u
        self.T = T

    def evaluate(self, t):
        nsteps = self.u.shape[0]
        idx = torch.floor((nsteps - 1) * t / self.T).to(torch.int64)
        return self.u[idx]

    def evaluate_tensor(self, t):
        nsteps = self.u.shape[0]
        idx = torch.floor((nsteps - 1) * t / self.T).to(torch.int64)
        return self.u[idx, :, :]

    def __call__(self, t, x, t_is_tensor=False):
        if not t_is_tensor:
            if len(self.evaluate(t).shape) == 2:
                evaluate_t = self.evaluate(t)
            else:
                evaluate_t = self.evaluate(t)[0, :, :]
            if len(x.shape) == 2:
                return torch.einsum("ij,bj->bi", evaluate_t, x)
            elif len(x.shape) == 3:
                return torch.einsum("ij,abj->abi", evaluate_t, x)
        else:
            if len(x.shape) == 2:
                return torch.einsum("bij,bj->bi", self.evaluate_tensor(t), x)
            elif len(x.shape) == 3:
                return torch.einsum("aij,abj->abi", self.evaluate_tensor(t), x)


class ConstantControlQuadratic:
    def __init__(self, pt, sigma, T, batchsize):
        self.pt = pt
        self.sigma = sigma
        self.T = T
        self.batchsize = batchsize

    def evaluate_pt(self, t):
        nsteps = self.pt.shape[0]
        idx = torch.floor(nsteps * t / self.T).to(torch.int64)
        return self.pt[idx]

    def __call__(self, t, x):
        control = -torch.matmul(
            torch.transpose(self.sigma, 0, 1), self.evaluate_pt(t)
        ).unsqueeze(0)
        return control


class ConstantControlLinear:
    def __init__(self, ut, T):
        self.ut = ut
        self.T = T

    def evaluate_ut(self, t):
        nsteps = self.ut.shape[0]
        idx = torch.floor(nsteps * t / self.T).to(torch.int64)
        return self.ut[idx]

    def evaluate_ut_tensor(self, t):
        nsteps = self.ut.shape[0]
        idx = torch.floor((nsteps - 1) * t / self.T).to(torch.int64)
        return self.ut[idx, :]

    def __call__(self, t, x, t_is_tensor=False):
        if not t_is_tensor:
            control = self.evaluate_ut(t).unsqueeze(0).repeat(x.shape[0], 1)
        else:
            control = self.evaluate_ut_tensor(t).unsqueeze(1).repeat(1, x.shape[1], 1)
        return control
    
class LowDimControl:
    def __init__(self, ut, T, xb, dim, delta_t, delta_x):
        self.ut = ut
        self.T = T
        self.xb = xb
        self.dim = dim
        self.delta_t = delta_t
        self.delta_x = delta_x

    def evaluate_ut(self, t, x):
        x_reshape = x.reshape(-1, self.dim)
        t = torch.tensor([t]).to(x.device)
        t = t.reshape(-1, 1).expand(x_reshape.shape[0], 1)
        tx = torch.cat([t, x_reshape], dim=-1)

        idx = torch.zeros_like(tx).to(tx.device).to(torch.int64)

        idx[:, 0] = torch.ceil(tx[:, 0] / self.delta_t).to(torch.int64)

        idx[:, 1:] = torch.floor((tx[:, 1:] + self.xb - self.delta_x) / self.delta_x).to(torch.int64)
        idx[:, 1:] = torch.minimum(
            idx[:, 1:],
            torch.tensor(self.ut.shape[1] - 1).to(torch.int64).to(idx.device),
        )
        idx[:, 1:] = torch.maximum(
            idx[:, 1:], torch.tensor(0).to(torch.int64).to(idx.device)
        )
        control = torch.zeros_like(x_reshape)
        for j in range(self.dim):
            idx_j = idx[:, [0, j + 1]]
            ut_j = self.ut[:, :, j]
            control_j = ut_j[idx_j[:, 0], idx_j[:, 1]]
            control[:, j] = control_j
        return control

    def evaluate_ut_tensor(self, t, x):
        t = t.reshape(-1, 1, 1).expand(x.shape[0], x.shape[1], 1)
        tx = torch.cat([t, x], dim=-1)
        tx_shape = tx.shape
        tx = tx.reshape(-1, tx.shape[2])

        idx = torch.zeros_like(tx).to(tx.device).to(torch.int64)

        idx[:, 0] = torch.ceil(tx[:, 0] / self.delta_t).to(torch.int64)

        idx[:, 1:] = torch.floor((tx[:, 1:] + self.xb - self.delta_x) / self.delta_x).to(torch.int64)
        idx[:, 1:] = torch.minimum(
            idx[:, 1:],
            torch.tensor(self.ut.shape[1] - 1).to(torch.int64).to(idx.device),
        )
        idx[:, 1:] = torch.maximum(
            idx[:, 1:], torch.tensor(0).to(torch.int64).to(idx.device)
        )
        control = torch.zeros_like(tx).to(tx.device)
        for j in range(self.dim):
            idx_j = idx[:, [0, j + 1]]
            ut_j = self.ut[:, :, j]
            control_j = ut_j[idx_j[:, 0], idx_j[:, 1]]
            control[:, j + 1] = control_j
        control = torch.reshape(control, tx_shape)[:, :, 1:]
        return control

    def __call__(self, t, x, t_is_tensor=False):
        if not t_is_tensor:
            return self.evaluate_ut(t, x)
        else:
            return self.evaluate_ut_tensor(t, x)

