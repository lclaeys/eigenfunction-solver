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
    
    def simulate_full(self, x, T, steps, save_every):
        """
        Simulate trajectories of Langevin dynamics, returning full trajectories.
        Args:
            x (tensor)[N,d]: initial conditions
            T (float): final time
            steps (int): number of timesteps (dt = T/steps)
            save_every (int): how often to save 
        Returns:
            samples (tensor)[steps/save_every+1,N,d]
        """
        assert steps % save_every == 0

        dt = T / (steps - 1)

        save_steps = steps // save_every

        samples = torch.zeros((save_steps+1,) + x.shape)
        samples[0] = x

        for i in tqdm(range(1,steps+1)):
            x = self.step(x, dt)
            if i % save_every == 0:
                samples[i // save_every] = x
        
        return samples
    
    def simulate_funcs(self, x, T, steps, funcs, save_every):
        """
        Simulate trajectories of Langevin dynamics, returning average of func at each timestep.
        (This is to save memory and not have to store the entire samples)
        Args:
            x (tensor)[N,d]: initial conditions
            T (float): final time
            steps (int): number of timesteps (dt = T/steps)
            funcs (list): list of batched functions
            save_every (int): how often to save
        Returns:
            fsamples (tensor)[len(funcs), steps+1]: average value of function at every timestep
        """
        assert steps % save_every == 0

        dt = T / (steps - 1)

        save_steps = steps // save_every

        fsamples = torch.zeros(len(funcs),save_steps+1)

        for j in range(len(funcs)):
            fsamples[j,0] = funcs[j](x).mean()
        for i in tqdm(range(1,steps+1)):
            x = self.step(x,dt)
            if i % save_every == 0:
                for j in range(len(funcs)):
                    fsamples[j,i//save_every] = funcs[j](x).mean()
        
        return fsamples

    def simulate_funcs_csv(self, x, T, steps, funcs, save_every, func_names, csv_file):
        """
        Simulate trajectories of Langevin dynamics, saving the average value of given functions at each timestep in a csv file under csv_path.
        Args:
            x (tensor)[N,d]: initial conditions
            T (float): final time
            steps (int): number of timesteps (dt = T/steps)
            funcs (list): list of batched functions
            save_every (int): how often to save
            func_names (list): names of functions (for file naming)
            csv_file (string): path to csv file where data will be saved
        """
        assert steps % save_every == 0

        dt = T / (steps - 1)

        save_steps = steps // save_every
        fsamples = torch.zeros(1+len(funcs))

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(['time'] + func_names)

            for j in range(len(funcs)):
                fsamples[1+j] = funcs[j](x).mean()
            
            writer.writerow(fsamples.tolist())

            for i in tqdm(range(1,steps+1)):
                x = self.step(x,dt)
                if i % save_every == 0:
                    fsamples[0] = i * dt
                    for j in range(len(funcs)):
                        fsamples[1+j] = funcs[j](x).mean()

                    writer.writerow(fsamples.tolist())

    def simulate_equilibrium(self, x, burnin_steps, dt, num_samples, sample_steps):
        """
        Simulate Langevin trajectory for long time in order to obtain samples from equilibrium.
        Args:
            x (tensor)[N,d]: initial conditions
            burnin_steps (int): how many burnin steps before starting to save samples
            dt (float): timestep
            num_samples (int): how many final samples are desired
            sample_steps (int): how many steps between saving samples (to reduce autocorrelation)
        """

        assert num_samples % x.shape[0] == 0

        samples_per_save = x.shape[0]
        total_saves = num_samples // samples_per_save

        samples = torch.zeros((num_samples, x.shape[1]))
        
        for i in range(burnin_steps):
            x = self.step(x,dt)

        samples[:samples_per_save] = x

        for j in tqdm(range(1,total_saves)):
            for i in range(sample_steps):
                x = self.step(x,dt)

            samples[j*samples_per_save:(j+1)*samples_per_save] = x

        return samples

