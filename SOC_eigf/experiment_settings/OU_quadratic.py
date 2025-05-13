"""

Model set up for quadratic case. Base code adapted from https://github.com/facebookresearch/SOC-matching.

Adapted to fit in our framework, and added exact solution for eigenfunctions of symmetric OU operator.

"""

import torch

import torch
import numpy as np
import heapq

from SOC_eigf_old2.method import NeuralSDE

from scipy.special import eval_hermitenorm
from scipy.special import factorial


class OU_Quadratic(NeuralSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        u=None,
        lmbd=1.0,
        sigma=torch.eye(2),
        A=torch.eye(2),
        P=torch.eye(2),
        Q=torch.eye(2),
        T = 1.0,
        method = "EIGF",
        eigf_cfg=None,
        ido_cfg=None
    ):
        super().__init__(
            device=device,
            dim=dim,
            u=u,
            lmbd=lmbd,
            sigma=sigma,
            T=T,
            method=method,
            eigf_cfg=eigf_cfg,
            ido_cfg=ido_cfg
        )
        self.A = A
        self.P = P
        self.Q = Q
        
        eigvals = torch.linalg.eigvalsh(A)
        if torch.all(eigvals > 0):
            self.confining = False
        elif torch.all(eigvals < 0):
            self.confining = True
        else:
            print('Warning: A is neither confining nor repulsive.')

        self.stored_coeff_matrix = torch.ones((1,1),device=device)
        self.stored_grad_coeff_matrix = torch.zeros((1,1),device=device)

    # Energy: b = -grad E
    def energy(self, x):
        return -0.5 * torch.sum(
            x * torch.einsum("ij,...j->...i", self.A, x), -1
        )

    # Base Drift
    def b(self, t, x):
        return torch.einsum("ij,...j->...i", self.A, x)

    # Gradient of base drift
    def nabla_b(self, t, x):
        if len(x.shape) == 2:
            return torch.transpose(self.A.unsqueeze(0).repeat(x.shape[0], 1, 1), 1, 2)
        elif len(x.shape) == 3:
            return torch.transpose(
                self.A.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1, 1),
                2,
                3,
            )
    
    # Laplacian of energy
    def Delta_E(self, x):
        return -torch.ones(x.shape[:-1],device=self.device) * torch.trace(self.A)

    # Running cost
    def f(self, t, x):
        return torch.sum(
            x * torch.einsum("ij,...j->...i", self.P, x), -1
        )

    # Gradient of running cost
    def nabla_f(self, t, x):
        return 2 * torch.einsum("ij,...j->...i", self.P, x)

    # Final cost
    def g(self, x):
        return torch.sum(
            x * torch.einsum("ij,...j->...i", self.Q, x), -1
        )

    # Gradient of final cost
    def nabla_g(self, x):
        return 2 * torch.einsum("ij,...j->...i", self.Q, x)
    
    # -------------- Exact solution ---------------

    def exact_eigvals(self, m):
        """
        Compute first m exact eigenvalues
        """
        A = -self.A

        if not torch.allclose(A, A.T):
            raise ValueError("Matrix A is not symmetric.")

        X = A.T @ A + 2 * self.P

        Lambda, U = torch.linalg.eigh(X)

        values, indices = self.smallest_combinations(Lambda, m)
        indices = torch.tensor(indices,device = self.device)

        return 1/self.lmbd * (-torch.trace(A) + (Lambda[None,:].sqrt() * (2*indices+1)).sum(dim=1))


    def exact_eigfunctions(self, x, m, use_scipy = False, return_grad = False):
        """
        Evaluate first m exact eigenfunctions of symmetric LQR at points x
        Args:
            sde (NeuralSDE): SDE of OU type
            x (tensor)[n,d]: evaluation points
            m (int): number of eigenfunctions to compute
            use_scipy (bool): whether to use scipy for evaluating hermite polynomials (not differentiable)
            return_grad (bool): whether to return gradient as well
        Returns:
            fx (tensor)[n,m]: first m eigenfunction evaluations
            (Optinal) grad_fx [n,m,d]: gradients of first m eigenfunctions
        """
        if x.requires_grad or return_grad:
            use_scipy = False
        
        # b = - grad E = - A
        A = -self.A

        if not torch.allclose(A, A.T):
            raise ValueError("Matrix A is not symmetric.")

        X = A.T @ A + 2 * self.P

        Lambda, U = torch.linalg.eigh(X)
        
        D =  1/self.lmbd * (-A + U @ torch.diag(Lambda.sqrt()) @ U.T)

        # QUADRATIC FORM
        quadratic_form = torch.exp(- 1/2 * torch.einsum('ij, ij -> i',x @ D, x))

        # HERMITE POLYNOMIAL TERM
        values, indices = self.smallest_combinations(Lambda, m)

        input_transform = U @ torch.pow(Lambda,1/4).diag() * np.sqrt(2/self.lmbd)
        reshaped_x = x @ input_transform

        if use_scipy:
            fx_poly = np.ones([x.shape[0],m])
            
            for i in range(m):
                # numpy
                hermite_evals = eval_hermitenorm(indices[i],reshaped_x.cpu().numpy())
                norms = np.sqrt(factorial(indices[i]))

                hermite_evals /= norms

                if len(hermite_evals.shape) != 1:
                    fx_poly[:,i] = np.prod(hermite_evals,axis=1)
        
            fx_poly = torch.tensor(fx_poly,dtype = x.dtype, device = x.device)

        else:
            fx_poly = torch.ones([x.shape[0],m], dtype = x.dtype, device = x.device)
            grad_fx_poly = torch.ones([x.shape[0],m,x.shape[1]], dtype = x.dtype, device = x.device)

            for i in range(m):
                # pytorch
                n = torch.tensor(indices[i], device = 'cpu') #indices on cpu
                if return_grad:
                    hermite_evals, grad_hermite_evals = self.eval_hermitenorm(n,reshaped_x, return_grad=True)
                else:
                    hermite_evals = self.eval_hermitenorm(n,reshaped_x, return_grad=False)

                norms = torch.sqrt(torch.tensor(factorial(indices[i]), dtype = x.dtype, device = x.device))

                hermite_evals /= norms

                fx_poly[:,i]  = torch.prod(hermite_evals,dim=1)

                if return_grad:
                    grad_hermite_evals /= norms

                    # (n,d)
                    ratio_term = grad_hermite_evals / hermite_evals
                    grad_fx_poly[:,i,:] = fx_poly[:,i,None] * (ratio_term) @ input_transform.T
        
        # TODO: add normalization
        normalization = (torch.prod(Lambda)**(-1/4) * torch.linalg.det(A.abs())**(1/2)) ** (1/2)

        fx = fx_poly * quadratic_form[:,None] / normalization

        if return_grad:
            grad_fx = quadratic_form[:,None,None] * (grad_fx_poly - (x @ D.T)[:,None,:] * fx_poly[:,:,None]) / normalization
            return fx, grad_fx
        return fx
    
    def exact_grad_log_gs(self, x):
        """
        Return grad log phi for the ground state.
        """
        A = -self.A

        if not torch.allclose(A, A.T):
            raise ValueError("Matrix A is not symmetric.")

        X = A.T @ A + 2 * self.P

        Lambda, U = torch.linalg.eigh(X)
        
        D =  1/self.lmbd * (-A + U @ torch.diag(Lambda.sqrt()) @ U.T)

        # QUADRATIC FORM
        return - torch.einsum('ni,ij->nj',x, D)
    
    def exact_eigf_control(self, ts, x, m, verbose=False):
        """
        Return control for symmetric LQR obtained from the first k eigenfunctions. Requires A,P,Q to be diagonal

        Args:
            ts (tensor)[N]: times
            x (tensor)[N,d] or [N,B,d]: system states
            m (int): number of eigf to use 
        """

        A = -self.A
        P = self.P
        Q = self.Q

        if not (torch.allclose(A.diag().diag(), A) and torch.allclose(P.diag().diag(), P)  and torch.allclose(Q.diag().diag(), Q)):
            raise ValueError("For exact control, all matrices must be diagonal.")
        
        X = A.T @ A + 2 * self.P

        Lambda, U = torch.linalg.eigh(X)
        
        D =  1/self.lmbd * (-A + U @ torch.diag(Lambda.sqrt()) @ U.T)

        original_shape = x.shape
        
        if len(original_shape)==3:
            x = x.reshape(-1,self.dim)

        # GS
        grad_log_gs = - torch.einsum('ni,ij->nj', x, D)

        if len(original_shape)==3:
            grad_log_gs = grad_log_gs.reshape(original_shape)

        # HIGHER ORDER TERMS
        values, indices = self.smallest_combinations(Lambda, m)
        eigvals = self.exact_eigvals(m)

        sigma_sq = 2*(A.diag()**2 + 2*P.diag()).sqrt() / (A.diag() + (A.diag()**2 + 2*P.diag()).sqrt() + 2*Q.diag())

        input_transform = U @ torch.pow(Lambda,1/4).diag() * np.sqrt(2/self.lmbd)
        reshaped_x = x @ input_transform

        grad_log_correction = torch.zeros_like(grad_log_gs, device=self.device)

        grad_term = torch.zeros_like(grad_log_gs, device=self.device)
        term = torch.zeros(grad_log_gs.shape[:2], device=self.device)

        for i in range(m):
            # compute i-th correction term.
            n = torch.tensor(indices[i], device = 'cpu') #indices on cpu
            n_gpu = torch.tensor(indices[i], device = self.device)

            if (n % 2 == 0).all():
                hermite_evals, grad_hermite_evals = self.eval_hermitenorm(n,reshaped_x, return_grad=True)

                norms = torch.sqrt(torch.tensor(factorial(indices[i]), dtype = x.dtype, device = x.device))

                hermite_evals /= norms

                fx_poly  = torch.prod(hermite_evals,dim=1)

                grad_hermite_evals /= norms

                # (n,d)
                ratio_term = grad_hermite_evals / hermite_evals
                grad_fx_poly = fx_poly[:,None] * (ratio_term) @ input_transform.T

                inner_prod_ratio = torch.prod((1/2*torch.lgamma(n_gpu+1) - torch.lgamma((n_gpu/2)+1)).exp() * ((sigma_sq-1)/2)**(n_gpu/2))
                exp_weight = torch.exp(-(eigvals[i]-eigvals[0]) * (self.T - ts) * self.lmbd/2)

                if verbose:
                    print(f"Adding term with alpha = {n}, inner product {inner_prod_ratio}, eigval {eigvals[i]}")

                if len(original_shape)==2:
                    term += exp_weight * inner_prod_ratio * fx_poly
                    grad_term += exp_weight[:,None] * inner_prod_ratio * grad_fx_poly
                    
                if len(original_shape)==3:
                    fx_poly = fx_poly.reshape(original_shape[:2])
                    grad_fx_poly = grad_fx_poly.reshape(original_shape)

                    term += exp_weight[:,None] * inner_prod_ratio * fx_poly
                    grad_term += exp_weight[:,None,None] * inner_prod_ratio * grad_fx_poly

        grad_log_correction = grad_term / term.unsqueeze(-1)
        
        if len(original_shape)==2:
            exact_control = self.lmbd * torch.einsum(
                    "ij,bj->bi",
                    torch.transpose(self.sigma, 0, 1),
                    grad_log_gs + grad_log_correction,
                )
            return exact_control
        
        if len(original_shape)==3:
            exact_control = self.lmbd * torch.einsum(
                    "ij,abj->abi",
                    torch.transpose(self.sigma, 0, 1),
                    grad_log_gs + grad_log_correction,
                )
            return exact_control


    def generate_probabilist_hermite_coeffs(self, m, dtype, device):
        """
        Generate the coefficients of the first m probabilist Hermite polynomials.
        
        Args:
            m (int): Number of Hermite polynomials to generate (non-negative integer).
            dtype
            device
        Returns:
            torch.Tensor: A 2D tensor of shape (m, m), where each row contains the 
                        coefficients of the corresponding Hermite polynomial, 
                        padded with zeros for alignment.
        """
        # Initialize a tensor to store coefficients
        coeffs = torch.zeros((m, m), dtype=dtype, device = device)
        
        # H_0(x) = 1
        if m > 0:
            coeffs[0, 0] = 1.0
        
        # H_1(x) = x
        if m > 1:
            coeffs[1, 1] = 1.0
        
        # Use the recurrence relation to compute higher-order coefficients
        for n in range(2, m):
            # H_n(x) = x * H_{n-1}(x) - (n-1) * H_{n-2}(x)
            coeffs[n, 1:] += coeffs[n-1, :-1]  # x * H_{n-1}(x)
            coeffs[n, :] -= (n - 1) * coeffs[n-2, :]  # - (n-1) * H_{n-2}(x)
        
        return coeffs


    def eval_hermitenorm(self, n, x, return_grad=False):
        """
        Implementation of scipy's eval_hermitenorm in a differentiable way using torch.pow.
        Args:
            n (tensor)[d]: degree of polynomial to evaluate at each dimension
            x (tensor)[N,d]: points to evaluate
            return_grad (bool): whether to return gradient as well
        Returns:
            He (tensor)[N,d]: values of He polynomial
            (Optional) grad_He (tensor)[N,d]: derivative of polynomial
        """
        m = n.max() + 1  # Maximum degree + 1

        # Generate probabilist Hermite coefficients
        if self.stored_coeff_matrix.shape[0] < m:
            self.stored_coeff_matrix = self.generate_probabilist_hermite_coeffs(m, dtype=x.dtype, device=x.device)
            self.stored_grad_coeff_matrix = torch.roll(self.stored_coeff_matrix,-1,1)
            self.stored_grad_coeff_matrix[:,-1] = 0
            self.stored_grad_coeff_matrix = self.stored_grad_coeff_matrix * torch.arange(1,m+1, device = x.device)[None,:]

        # (d, m): Extract the relevant coefficients for the input degrees
        coeff_matrix = self.stored_coeff_matrix[:m,:m]
        grad_coeff_matrix =  self.stored_grad_coeff_matrix[:m,:m]
        coeffs = coeff_matrix[n]
        grad_coeffs = grad_coeff_matrix[n]

        # Compute powers of x
        x_powers = torch.stack([torch.pow(x, i) for i in range(m)], dim=0)  # Shape: (m, N, d)

        # Compute Hermite polynomial values using einsum
        if return_grad:
            return torch.einsum('ijk,ki->jk', x_powers, coeffs), torch.einsum('ijk,ki->jk', x_powers, grad_coeffs)
        
        return torch.einsum('ijk,ki->jk', x_powers, coeffs)

    # Helper function to compute which multi-indices correspond to smallest eigenvalues
    @staticmethod
    def smallest_combinations(x, m):
            """
            Args:
                x (tensor)[d]: input array of eigenvalues of A
                m (int) 
            Returns:
                vals (tensor)[m]: smallest linear combinations of eigenvalues
                vecs (ndarray)[m,d]: indices of those combinations
            """
            n = len(x)
            
            heap = []
            visited = set()
            
            # Start with the combination (0, 0, ..., 0)
            initial = (0, [0] * n)  # (S, a_vector)
            heapq.heappush(heap, initial)
            visited.add(tuple([0] * n))
            
            vals = []
            vecs = []
            
            while len(vals) < m:
                S, a_vector = heapq.heappop(heap)
                vals.append(S) 
                vecs.append(a_vector)

                # Generate new combinations
                for i in range(n):
                    new_a_vector = a_vector[:]
                    new_a_vector[i] += 1
                    new_S = S + x[i]
                    
                    if tuple(new_a_vector) not in visited:
                        heapq.heappush(heap, (new_S, new_a_vector))
                        visited.add(tuple(new_a_vector))
            
            return torch.tensor(vals,device=x.device), np.array(vecs)