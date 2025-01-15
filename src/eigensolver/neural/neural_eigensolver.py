import torch.nn
import torch
import importlib

from src.eigensolver.base_eigensolver import BaseSolver
from src.eigensolver.neural.loss.orth_loss import BasicOrthogonalityLoss, CovOrthogonalityLoss
from src.eigensolver.neural.loss.variational_loss import VariationalLoss
from torch.utils.data import Dataset, DataLoader

class NeuralSolver(BaseSolver):
    """
    Solver based on training a neural network to minimize regularized variational loss.
    """
    def __init__(self, energy, samples, model, optimizer, params, scheduler = None, *args, **kwargs):
        """
        Args:
            energy (BaseEnergy): energy function
            samples (tensor): samples from the stationary distribution
            model (nn.Module): neural network model
            optimizer (torch.optim.Optimizer): optimizer for training the model
            scheduler (optional): scheduler for training
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
        
        self.model_device = params.get('model_device','cuda')
        self.base_device = 'cpu'
        self.verbose = params.get('verbose',False)
        self.beta = params.get('beta', 1.0)
        
        self.k = params.get('k', 1)
        if self.beta is not torch.tensor:
            self.beta = torch.ones(self.k)*self.beta

        self.num_samples = params.get('num_samples',10000)
        self.batch_size = params.get('batch_size',5000)
        self.pca_reg = params.get('pca_reg', 1e-6)
        self.dim = energy.dim
        self.scheduler = scheduler

        # put model on device
        self.model = model.to(self.model_device)

        self.energy = energy
        self.optimizer = optimizer

        # TODO: add flexibility here
        self.var_loss = VariationalLoss(self.beta)
        self.orth_loss = BasicOrthogonalityLoss()

        self.samples = samples
        self.dataloader = DataLoader(
                            samples,
                            batch_size=self.batch_size,
                            shuffle=True,
                            pin_memory=True
                        )
        
        self.rotation = None
        self.laplacian = None
        

    # TODO: maybe add .fit() method ?

    def _compute_loss(self, model, x):
        """
        Compute the loss, given by beta*sum_i <grad f_i, grad_fi>_mu + orth_loss

        Args:
            model (nn.Module): model
            x (torch.tensor on model_device): tensor on same device as model
        
        Returns:
            loss (torch.tensor on model_device)
        """

        grad_outputs = torch.eye(self.k, device = self.model_device)[:,None,:].expand([self.k,x.shape[0],self.k])
        
        fx = model(x)
        grad_fx = torch.autograd.grad(outputs = fx, inputs = x, grad_outputs = grad_outputs, is_grads_batched=True, create_graph=True)[0].transpose(0,1)
        
        loss_1 = self.var_loss(grad_fx)  # Variational loss
        loss_2 = self.orth_loss(fx)      # Orthogonal loss

        return loss_1 + loss_2
        
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            loss (tensor on cpu)
        """

        for batch_idx, batch in enumerate(self.dataloader):
            batch = batch.to(self.model_device)
            batch = batch.requires_grad_()
            self.optimizer.zero_grad()  # Clear gradients

            loss = self._compute_loss(self.model, batch)

            loss.backward()  # Backward pass
            self.optimizer.step()  # Update model parameters
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.detach().cpu()
    
    def compute_eigfuncs(self):
        """
        Performs PCA to compute eigenfunctions and eigenvalues under the assumption of convergence.
        
        The rotation that is to be applied to the outputs of the network is saved under self.rotation (on CPU)
        The eigenvalues obtained from this are saved under self.eigvals
        """
        with torch.no_grad():
            x_pca = self.samples[:self.num_samples].to(self.model_device)
            fx_pca = self.model(x_pca)[:,1:].to('cpu')

            # higher precision for eigh
            fx_pca = fx_pca.double()
            
            cov = torch.sum(fx_pca[:,:,None]*fx_pca[:,None,:],dim=0)/fx_pca.size(0)
            
            error = torch.linalg.eigvalsh(cov)[0]
            if error < 0:
                self.pca_reg += -error*1.1

            cov = cov + self.pca_reg*torch.eye(cov.shape[0],dtype=torch.float64)

            D, U = torch.linalg.eigh(cov)
            D, U = D.float().to(self.base_device), U.float().to(self.base_device)

            self.eigvals = torch.zeros(self.k)
            self.eigvals[1:] = 2/self.beta[1:]*(1-D)
            self.rotation = U@torch.diag(D**(-1/2))
            self.indices = torch.argsort(self.eigvals)
            self.eigvals = self.eigvals[self.indices]

    def predict(self,x):
        """
        Evaluate learned eigenfunction at points x.

        Args:
            x (tensor)[n,d]: points at which to evaluate
        Returns:
            fx (tensor)[n,m]: learned eigenfunctions evaluated at points x.
        """
        with torch.no_grad():
            if self.rotation is None:
                self.compute_eigfuncs()

            x = x.to(self.model_device)
            outputs = self.model(x).to(self.base_device)

            outputs[:, 1:] = outputs[:, 1:]@self.rotation

            return outputs[:, self.indices]
    
    def predict_grad(self, x):
        """
        Evaluate gradient of learned eigenfunction at points x.

        Args:
            x (tensor)[n,d]: points at which to evaluate
        Returns:
            grad_fx (tensor)[n,m,d]: gradient of learned eigenfunctions evaluated at points x.
        """
        # TODO Check this implementation
        with torch.no_grad():
            x = x.to(self.model_device)

            def func(x):
                return self.model(x)

            grad = torch.torch.func.jacrev(func)
            batch_grad = torch.func.vmap(grad)

            # (n,m,d)
            grad_outputs = batch_grad(x).to(self.base_device)

            grad_outputs[:,1:,:] = (grad_outputs[:,1:,:].transpose(1,2) @ self.rotation).transpose(1,2)

            return grad_outputs[:,self.indices,:]
    
    def predict_laplacian(self, x):
        """
        Evaluate laplacian of learned eigenfunction at points x.

        Args:
            x (ndarray or tensor)[n,d]: points at which to evaluate
        Returns:
            delta_fx (ndarray)[n,m]: laplacian of learned eigenfunctions evaluated at points x.
        """
        def func(x):
            return self.model(x)

        hessian = torch.func.hessian(func)
        laplacian = torch.vmap(lambda x: torch.diagonal(hessian(x),dim1=1,dim2=2).sum(dim=1))
        
        with torch.no_grad():
            x = x.to(self.model_device)
            laplacian_outputs =  laplacian(x).to(self.base_device)

            laplacian_outputs[:,1:] = laplacian_outputs[:,1:]@self.rotation
            
            return laplacian_outputs[:, self.indices]
    
    def predict_Lf(self, x):
        """
        Evaluate Lf of learned eigenfunction at points x.

        Args:
            x (ndarray)[n,d]: points at which to evaluate
        Returns:
            Lfx (ndarray)[n,m]: Lf evaluated at points x.
        """

        energy_grad = self.energy.grad(x)

        Lfx = -self.predict_laplacian(x) + torch.bmm(self.predict_grad(x), energy_grad.unsqueeze(2)).squeeze(2)

        return Lfx
    
    def fit_eigvals(self, x):
        """
            predict eigvals using OLS
        """
        
        self.fx = self.predict(x)
        self.Lfx = self.predict_Lf(x)

        self.fitted_eigvals = torch.sum(self.fx*self.Lfx,dim=0)/torch.sum(self.fx**2,dim=0)
        return self.fitted_eigvals