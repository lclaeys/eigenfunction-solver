import torch
import math
from tqdm import tqdm

class LangevinSampler():
    """
    Class for sampling from the Langevin dynamics with a given energy E.
    """
    def __init__(self, energy):
        """"
        Args:
            energy (BaseEnergy): energy function
        """
        self.energy = energy
        self.dim = self.energy.dim

    def step(self, x, dt):
        """
        Simulate 1 step of Langevin dynamics using Forward Euler.

        Args:
            x (tensor)[N,d] : initial points
            dt (float): timestep

        Returns:
            new_x (tensor)[N,d]: new points
        """
        grad_E = self.energy.grad(x)
        x.requires_grad_(False)
        dW = torch.randn(x.shape)

        new_x = x - grad_E * dt + math.sqrt(2 * dt) * dW

        return new_x
    
    def simulate_full(self, x, T, steps):
        """
        Simulate trajectories of Langevin dynamics, returning full trajectories.
        Args:
            x (tensor)[N,d]: initial conditions
            T (float): final time
            steps (int): number of timesteps (dt = T/steps)

        Returns:
            samples (tensor)[steps+1,N,d]
        """
        dt = T / (steps - 1)

        samples = torch.zeros((steps+1,) + x.shape)
        samples[0] = x

        for i in tqdm(range(1,steps+1)):
            x = self.step(x, dt)
            samples[i] = x
        
        return samples
    
    def simulate_func(self, x, T, steps, func):
        """
        Simulate trajectories of Langevin dynamics, returning average of func at each timestep.
        (This is to save memory and not have to store the entire samples)
        Args:
            x (tensor)[N,d]: initial conditions
            T (float): final time
            steps (int): number of timesteps (dt = T/steps)
            func: batched function
        Returns:
            fsamples (tensor)[steps+1]: average value of function at every timestep
        """
        dt = T / (steps - 1)
        fsamples = torch.zeros(steps+1)

        fsamples[0] = func(x).mean()
        for i in tqdm(range(1,steps+1)):
            x = self.step(x,dt)
            fsamples[i] = func(x).mean()
        
        return fsamples
    
    def simulate(self, x, T, steps):
        """
        Simulate trajectories of Langevin dynamics, only returning final states
        Args:
            x (tensor)[N,d]: initial conditions
            T (float): final time
            steps (int): number of timesteps (dt = T/steps)

        Returns:
            samples (tensor)[steps+1,N,d]
        """
        dt = T / steps - 1

        for i in tqdm(range(1,steps+1)):
            x = self.step(x, dt)
        
        return x


