import torch
import math
from tqdm import tqdm
import csv

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
    
    def simulate_funcs(self, x, T, steps, funcs):
        """
        Simulate trajectories of Langevin dynamics, returning average of func at each timestep.
        (This is to save memory and not have to store the entire samples)
        Args:
            x (tensor)[N,d]: initial conditions
            T (float): final time
            steps (int): number of timesteps (dt = T/steps)
            funcs (list): list of batched functions
        Returns:
            fsamples (tensor)[len(funcs), steps+1]: average value of function at every timestep
        """
        dt = T / (steps - 1)
        fsamples = torch.zeros(len(funcs),steps+1)

        for j in range(len(funcs)):
            fsamples[j,0] = funcs[j](x).mean()
        for i in tqdm(range(1,steps+1)):
            x = self.step(x,dt)
            for j in range(len(funcs)):
                fsamples[j,i] = funcs[j](x).mean()
        
        return fsamples

    def simulate_funcs_csv(self, x, T, steps, funcs, func_names, csv_file):
        """
        Simulate trajectories of Langevin dynamics, saving the average value of given functions at each timestep in a csv file under csv_path.
        Args:
            x (tensor)[N,d]: initial conditions
            T (float): final time
            steps (int): number of timesteps (dt = T/steps)
            funcs (list): list of batched functions
            func_names (list): names of functions (for file naming)
            csv_file (string): path to csv file where data will be saved
        """
        dt = T / (steps - 1)
        fsamples = torch.zeros(len(funcs),steps+1)
        times = torch.linspace(0,T,steps+1)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(['time'] + func_names)

            for j in range(len(funcs)):
                fsamples[j,0] = funcs[j](x).mean()
            
            writer.writerow([0] + list(fsamples[:,0]))
            for i in tqdm(range(1,steps+1)):
                x = self.step(x,dt)
                for j in range(len(funcs)):
                    fsamples[j,i] = funcs[j](x).mean()

                writer.writerow([times[i]] + list(fsamples[:,i]))
    
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


