import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

project_root = os.path.abspath("..")  # If notebooks is one folder above src
two_levels_up = os.path.abspath("../..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if two_levels_up not in sys.path:
    sys.path.insert(0,two_levels_up)
# Get the parent directory (one level above)
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)

import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch_optimizer as torch_optim

from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

# Import wandb
import wandb    

from tqdm import tqdm

from src.energy.quadratic import QuadraticEnergy

from src.eigensolver.neural.network.feedforward import FeedForwardNetwork, ConstantFFN, ConstantLinearFFN


def compute_D(A,P,lmbda):
    assert torch.allclose(A, torch.eye(A.shape[0],device=A.device)*A[0,0])
    Lambda, U = torch.linalg.eigh(P)

    a = torch.diag(A)
    sign = torch.where(a > 0, 1.0, -1.0)
    kappa = torch.diag(A) / lmbda * (1 + sign * torch.sqrt(1 + 2 / a**2 * Lambda))

    return U @ torch.diag(kappa) @ U.T

def exact_eigfunctions(x, m, params, return_grad = False):
    """
    Evaluate first m exact eigenfunctions of K at points x
    Args:
        x (tensor)[n,d]: evaluation points
        m (int): number of eigenfunctions to compute
        params (dict): problem parameters
        return_grad (bool): whether to return gradient
    Returns:
        fx (tensor)[n,m]: first m eigenfunction evaluations
        (Optinal) grad_fx (tensor)[n,m,d]: gradients of first m eigenfunctions
    """
    A = params['A']
    P = params['P']
    dim = A.shape[0]
    lmbda = params['lmbda']

    D = compute_D(A, P, lmbda)
    energy = QuadraticEnergy(-2/lmbda * A + 2*D)

    normalizer = (torch.linalg.det(-2/lmbda*A)** (1/2) / torch.linalg.det(-2/lmbda * A + 2 * D) ** (1/2)) ** (1/2)
    quadratic_form = torch.exp(- 1/2 * torch.einsum('ij, ij -> i',x @ D, x))
    
    if return_grad:
        wx, grad_wx = energy.exact_eigfunctions(x, m, use_scipy=False, return_grad=True)
        fx = wx * quadratic_form[:,None] / normalizer
        grad_fx = (grad_wx - (x @ D.T)[:,None,:] * wx[:,:,None]) * quadratic_form[:, None, None] / normalizer

        return fx, grad_fx
    
    else:
        wx = energy.exact_eigfunctions(x, m, use_scipy=False)
        fx = wx * quadratic_form[:,None] / normalizer
        
        return fx

def exact_eigvals(k, params, shift = False):
    """
    Compute m smallest exact eigenvalues of K
    Args:
        m (Int)
        shift (bool): whether to include constant shift
    Returns:
        eigvals (tensor)
    """
    lmbda = params['lmbda']
    
    A = params['A']
    P = params['P']

    D = compute_D(A, P, lmbda)
    
    energy = QuadraticEnergy(-2/lmbda * A + 2*D)
    
    if shift:
        eigvals = lmbda/2 * (energy.exact_eigvals(k) + torch.trace(D))
    else:
        eigvals = lmbda/2 * (energy.exact_eigvals(k))

    return eigvals

def R_func(x, params):
    """
    The function R in Kf = Lf + Rf.
    Args:
        x (tensor)[N,d]: input values
        params (dict): problem parameters
    Returns:
        Rx (tensor)[N]: output values
    """
    return 2/params['lmbda']**2 * torch.einsum('ij, ij -> i',x @ params['P'], x)

def compute_loss(model, x, params):
    """
    Compute loss of the model give samples x
    Args:
        model (nn.Module): model
        x (torch.tensor): samples
        params (problem parameters)
    Returns:
        loss (tensor)
    """
    fx = model(x)
    Rx = R_func(x, params)
    grad_fx = torch.autograd.grad(outputs = fx,
                              inputs = x, 
                              grad_outputs=torch.ones_like(fx),
                              retain_graph=True,
                              create_graph=True)[0]
    x.detach_()
    
    sq_norm = torch.mean(fx[:,0]**2)
    sq_grad_norm = torch.mean(torch.norm(grad_fx,dim=1,p=2)**2)
    R_norm = torch.mean(fx[:,0]**2 * Rx)

    var_loss = R_norm + sq_grad_norm
    orth_loss = (sq_norm - 1)**2

    return var_loss + 10*orth_loss, var_loss, orth_loss

def ritz_loss(model, x, params):
    """
    Compute loss of the model give samples x
    Args:
        model (nn.Module): model
        x (torch.tensor): samples
        params (problem parameters)
    Returns:
        loss (tensor)
    """
    fx = model(x)
    Rx = R_func(x, params)
    grad_fx = torch.autograd.grad(outputs = fx,
                              inputs = x, 
                              grad_outputs=torch.ones_like(fx),
                              retain_graph=True,
                              create_graph=True)[0]
    
    x.detach_()

    sq_norm = torch.mean(fx[:,0]**2)
    sq_grad_norm = torch.mean(torch.norm(grad_fx,dim=1,p=2)**2)
    R_norm = torch.mean(fx[:,0]**2 * Rx)

    loss = (sq_grad_norm + R_norm) / sq_norm

    return loss

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



dim = 20
A= -torch.eye(dim) * 0.2
P = torch.eye(dim) * 0.2
Q = torch.eye(dim) * 0.1
sigma = torch.eye(dim)
lmbda = torch.tensor(1.0)

params = {'A': A, 'P': P, 'Q': Q, 'sigma': sigma, 'lmbda': lmbda}

equilibrium_energy = QuadraticEnergy(-A * 2 / lmbda)
num_samples =  2**17
samples = equilibrium_energy.exact_sample((num_samples,))

# Hyperparameters
num_epochs = 40000
batch_size = 2**17

# Define your model architecture
model_arch = [dim, 256,512,1024,1024,512,256,1]

exact_fx = exact_eigfunctions(samples, 1, params)

def setup(rank, world_size):
    """Initialize process group for distributed training."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Destroy process group to free resources."""
    dist.destroy_process_group()

def train(rank, world_size):
    """Training function for each GPU process."""
    setup(rank, world_size)

    # Set device for the current process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.init()
    torch.cuda.set_device(device)

    if rank == 0:
        wandb.init(
            project="neural-eigenfunction-learner",
            name = 'nonritz_test_20d_unet_big',
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "model_arch": model_arch,
                "learning_rate": 1e-4,
            }
        )
    else:
        wandb.init(mode="disabled")

    # Convert parameters to GPU
    cuda_params = {key: params[key].to(device) for key in params}

    # Define model
    # Scale Up to ~1M Parameters
    input_size = dim
    hidden_size = 1024  # Large hidden size
    output_size = 1
    num_layers = 4

    # Create model
    model = FullyConnectedUNet(dim=dim,k=1,hdims=[512,256,128]).to(device)
    ddp_model = DDP(model, device_ids=[rank])
    
    for param in ddp_model.parameters():
        param.data = param.data.contiguous()

    # Register a hook to force each gradient to be contiguous.
    for param in ddp_model.parameters():
        if param.requires_grad:
            param.register_hook(lambda grad: grad.contiguous())

    # Optimizer with base learning rate
    base_lr = 1e-4
    optimizer = optim.Adam(params=ddp_model.parameters(), lr=base_lr)

    # Scheduler: Warm-up for the first epochs then cosine annealing for the rest.
    warmup_epochs = 2000
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda epoch: ((epoch + 1) / warmup_epochs) if epoch < warmup_epochs else
                                  0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    )

    # Local progress bar (only enable on rank 0)
    progress_bar = tqdm(range(num_epochs), desc=f"Rank {rank} Training", leave=True, position=rank, disable=(rank != 0))
    err = 1.0

    for epoch in progress_bar:
        ddp_model.train()
        optimizer.zero_grad()

        # Sample batch from distributed sampler
        torch.manual_seed(world_size * epoch)

        # generate samples
        batch = equilibrium_energy.exact_sample((batch_size,)).to(device)

        # loaded samples
        # random_indices = torch.randint(0, num_samples, (batch_size,))
        # batch = samples[random_indices].to(device)

        batch.requires_grad_()

        # Compute loss
        loss, var_loss, orth_loss = compute_loss(ddp_model, batch, cuda_params)
        #loss = ritz_loss(model, batch, cuda_params)
        loss.backward()
        optimizer.step()

        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}",
                                       "error": f"{err:.3e}"})
            wandb.log({
                "epoch": epoch,
                "loss": loss.item(),
                "error": err,
                "var_loss": var_loss.item(),
                "orth_loss": orth_loss.item(),
                "lr": current_lr
            })

        #Evaluate every 10 epochs
        if epoch % 10 == 0 and rank == 0:
            ddp_model.eval()
            with torch.no_grad():
                fx = ddp_model(batch).to('cpu')
                exact_fx = exact_eigfunctions(batch,1,cuda_params).to('cpu')
                norm = torch.mean(fx[:,0]**2).sqrt()
                err = min(((fx / norm - exact_fx) ** 2).mean(), ((fx / norm + exact_fx) ** 2).mean())
        # save every 10k epochs
        if (epoch + 1) % 10000 == 0 and rank == 0:
            torch.save(ddp_model.state_dict(), f"model_weights3_{epoch+1}.pth")
        
        scheduler.step()
    
    cleanup()  # Cleanup process group after training
    if rank == 0:
        wandb.finish()
        

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)