import torch.nn
import torch
import importlib
import numpy as np
from scipy.linalg import eigh

from src.eigensolver.base_eigensolver import BaseSolver
from src.eigensolver.neural.loss.orth_loss import BasicOrthogonalityLoss, CovOrthogonalityLoss
from src.eigensolver.neural.loss.variational_loss import VariationalLoss
from torch.utils.data import Dataset, DataLoader

class NeuralSolver(BaseSolver):
    """
    Solver based on training a neural network to minimize regularized variational loss.
    """
    def __init__(self, energy, samples, model, optimizer, params, *args, **kwargs):
        """
        Args:
            energy (BaseEnergy): energy function
            samples (ndarray): samples from the stationary distribution
            model (nn.Module): neural network model
            optimizer (torch.optim.Optimizer): optimizer for training the model
            params (dict): additional solver parameters:
                device (torch.device): device on which to train the model
                verbose (bool): whether to print additional output
                beta (float): regularization parameter. Loss is beta*var_loss + orth_loss
                k (int): number of eigenfunctions to compute
                batch_size (int): batch size used during training
                pca_reg (float): regularizer added to covariance matrix for PCA
                num_samples (int): number of samples used to estimate expectation for PCA
        """
        super().__init__(energy, *args, **kwargs)
        
        self.device = params.get('device','cpu')
        self.verbose = params.get('verbose',False)
        self.beta = params.get('beta', 1)
        self.k = params.get('k', 1)
        self.num_samples = params.get('num_samples',10000)
        self.batch_size = params.get('batch_size',5000)
        self.pca_reg = params.get('pca_reg', 1e-6)
        self.dim = energy.dim

        # convert samples to tensor
        self.samples = torch.tensor(samples,dtype=torch.float32)

        # put model on device
        self.model = model.to(self.device)

        self.energy = energy
        self.optimizer = optimizer

        # TODO: add flexibility here
        self.var_loss = VariationalLoss()
        self.orth_loss = BasicOrthogonalityLoss()

        self.dataloader = DataLoader(
                            samples,
                            batch_size=self.batch_size,
                            shuffle=True,
                            pin_memory=True
                        )
        
        self.rotation = None
        

    # TODO: maybe add .fit() method ?

    def _compute_loss(self, model, x):
        """
        Compute the loss, given by beta*sum_i <grad f_i, grad_fi>_mu + orth_loss

        Args:
            model (nn.Module): model
            x (torch.tensor): tensor on same device as model
        
        Returns:
            loss (torch.tensor)
        """

        grad_outputs = torch.eye(self.k, device = self.device)[:,None,:].expand([self.k,x.shape[0],self.k])
        
        fx = model(x)
        grad_fx = torch.autograd.grad(outputs = fx, inputs = x, grad_outputs = grad_outputs, is_grads_batched=True, create_graph=True)[0].transpose(0,1)
        
        loss_1 = self.var_loss(grad_fx)  # Variational loss
        loss_2 = self.orth_loss(fx)      # Orthogonal loss

        return self.beta*loss_1 + loss_2
        
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            loss (ndarray)
        """

        for batch_idx, batch in enumerate(self.dataloader):
            batch = batch.to(self.device).to(dtype=torch.float32)
            batch = batch.requires_grad_()
            self.optimizer.zero_grad()  # Clear gradients

            loss = self._compute_loss(self.model, batch)

            loss.backward()  # Backward pass
            self.optimizer.step()  # Update model parameters
        
        return loss.detach().cpu().numpy()
    
    def compute_eigfuncs(self):
        """
        Performs PCA to compute eigenfunctions and eigenvalues under the assumption of convergence.
        
        The rotation that is to be applied to the outputs of the network is saved under self.rotation (on CPU)
        The eigenvalues obtained from this are saved under self.eigvals
        """
        with torch.no_grad():
            x_pca = self.samples[:self.num_samples].to(self.device)
            fx_pca = self.model(x_pca)[:,1:]

            # higher precision for eigh
            fx_pca = np.array(fx_pca.to('cpu'),dtype=np.float64)
            
            cov = np.sum(fx_pca[:,:,None]*fx_pca[:,None,:],axis=0)/fx_pca.shape[0]
            
            error = eigh(cov, eigvals_only=True, subset_by_index=[0, 0])[0]
            if error < 0:
                self.pca_reg += -error*1.1

            cov = cov + self.pca_reg*np.eye(cov.shape[0])

            D, U = np.linalg.eigh(cov)
            self.eigvals = np.zeros(self.k)
            self.eigvals[1:] = 2/self.beta*(1-D)
            self.rotation = U@np.diag(D**(-1/2))
            self.indices = np.argsort(self.eigvals)
            self.eigvals = self.eigvals[self.indices]

            # TODO implement fitting of eigvals. For now this does not yet work because predict_Lfx does not work.
            self.fitted_eigvals = self.eigvals

    def predict(self,x):
        """
        Evaluate learned eigenfunction at points x.

        Args:
            x (ndarray or tensor)[n,d]: points at which to evaluate
        Returns:
            fx (ndarray)[n,m]: learned eigenfunctions evaluated at points x.
        """
        with torch.no_grad():
            if self.rotation is None:
                self.compute_eigfuncs()

            x = torch.tensor(x,device = self.device, dtype=torch.float32)
            outputs = self.model(x)

            outputs = np.array(outputs.to('cpu'),dtype=np.float64)
            outputs[:, 1:] = outputs[:, 1:]@self.rotation

            return outputs[:, self.indices]
    
    def predict_grad(self, x):
        """
        Evaluate gradient of learned eigenfunction at points x.

        Args:
            x (Tensor)[n,d]: points at which to evaluate
        Returns:
            grad_fx (Tensor)[n,m,d]: gradient of learned eigenfunctions evaluated at points x.
        """
        # TODO Check this implementation

        x = torch.tensor(x)
        x = x.requires_grad_()
        grad_outputs = torch.eye(self.k)[:,None,:].expand([self.k,x.shape[0],self.k])
        
        outputs = self.model(x)
        grad_outputs = torch.autograd.grad(outputs = outputs, inputs = x, grad_outputs = grad_outputs, is_grads_batched=True, create_graph = False)[0].transpose(0,1)
        
        grad_outputs[:,1:,:] = torch.einsum('nmd, mm -> nmd', grad_outputs[:,1:,:], self.rotation)

        return grad_outputs[:,self.indices,:].detach().numpy()
    
    def predict_laplacian(self, x):
        """
        Evaluate laplacian of learned eigenfunction at points x.

        Args:
            x (Tensor)[n,d]: points at which to evaluate
        Returns:
            delta_fx (Tensor)[n,m]: laplacian of learned eigenfunctions evaluated at points x.
        """
        # TODO
        raise NotImplementedError
        
    
    def predict_Lf(self, x):
        """
        Evaluate Lf of learned eigenfunction at points x.

        Args:
            x (Tensor)[n,d]: points at which to evaluate
        Returns:
            Lfx (Tensor)[n,m]: Lf evaluated at points x.
        """
        # TODO
        raise NotImplementedError
    
    def fit_eigvals(self, x):
        """
            predict eigvals using OLS
        """
        self.fx = self.predict(x)
        self.Lfx = self.predict_Lf(x)

        self.fitted_eigvals = np.sum(self.fx*self.Lfx,axis=0)/np.sum(self.fx**2,axis=0)
        return self.fitted_eigvals