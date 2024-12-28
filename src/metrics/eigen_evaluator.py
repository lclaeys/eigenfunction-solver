import numpy as np
import matplotlib.pyplot as plt

class EigenEvaluator():
    def __init__(self, energy):
        self.energy = energy
        self.exact_eigfuncs = None

    def evaluate_metrics(self, solver, x, metrics = None, k = 1):
        """
        returns dict with k-dim arrays of cost for first k eigenfunctions
        """
        out = {}
        fx = None
        Lfx = None
        grad_fx = None

        for metric in metrics:
            if metric == "eigen_error":
                """
                MSE of Lf(x) - f(x)
                """
                if fx is None:
                    fx = solver.predict(x)[:,:k]
                if Lfx is None:
                    Lfx = solver.predict_Lf(x)[:,:k]
                errs = np.mean((solver.eigvals[:k]*fx-Lfx)**2,axis=0)
                out[metric] = errs

            if metric == "orth_error":
                """
                MSE \|f(x)f(x)^T -  diag(f(x)f(x)^T)\|
                """
                if fx is None:
                    fx = solver.predict(x)[:,:k]
                cov = fx.T@fx/x.shape[0]
                cov_err = cov - np.eye(cov.shape[0])
                errs = np.array([np.mean(cov_err[:i,:i]**2) for i in range(1,k)])
                err_diffs = errs - np.roll(errs,1)
                err_diffs[0] = errs[0]
                out[metric] = err_diffs

            if metric == "eigen_cost":
                """
                sum_i^k <grad f_i , grad f_i >
                """
                if grad_fx is None:
                    grad_fx = solver.predict_grad(x)[:,:k,:]
                costs = np.diag(np.sum(np.matmul(grad_fx,np.transpose(grad_fx,axes=[0,2,1])),axis=0))
                out[metric] = costs

            if metric == "eigenvalue_mse":
                eigvals = self.energy.exact_eigvals(k)
                errs = (eigvals[:k]-solver.eigvals[:k])**2
                out[metric] = errs

            if metric == "eigenfunc_mse":
                if fx is None:
                    fx = solver.predict(x)[:,:k]
                i = 0
                eigvals = self.energy.exact_eigvals(k)
                if self.exact_eigfuncs is None:
                    self.exact_eigfuncs =  self.energy.exact_eigfunctions(x, k)
                eigfuncs = self.exact_eigfuncs
                errs = np.zeros(k)
                rotated_fx = np.zeros_like(fx)
                while i < k:
                    cur_eigval = eigvals[i]
                    e = 1
                    j = 1
                    while i + j < k and eigvals[i+j] == cur_eigval:
                        e += 1
                        j += 1
                    # i = first index of eigenspace
                    # i + j - 1 = last index of eigenspace
                    # e = dimension of eigenspace
                    F = eigfuncs[:,i:i+j]
                    Fhat = fx[:,i:i+j]

                    R = self.solve_procrustes(Fhat, F)
                    rotated_Fhat = Fhat@R
                    err = np.mean((rotated_Fhat - F)**2,axis=0)
                    
                    errs[i:i+j] = err
                    rotated_fx[:,i:i+j] = rotated_Fhat

                    i = i + j

                self.rotated_fx = rotated_fx
                out[metric] = errs
        return out
    
    def plot_eigfuncs(self, solver, x, k):
        if solver.dim == 1:
            tmin, tmax = np.quantile(x, [0.1,0.9])
            t = np.linspace(tmin, tmax,1000)[:,None]
            
            fx = solver.predict(t)[:,:k]
            i = 0
            eigvals = self.energy.exact_eigvals(k)
            eigfuncs =  self.energy.exact_eigfunctions(t, k)
            rotated_fx = np.zeros_like(fx)
            while i < k:
                cur_eigval = eigvals[i]
                e = 1
                j = 1
                while i + j < k and eigvals[i+j] == cur_eigval:
                    e += 1
                    j += 1
                # i = first index of eigenspace
                # i + j - 1 = last index of eigenspace
                # e = dimension of eigenspace
                F = eigfuncs[:,i:i+j]
                Fhat = fx[:,i:i+j]

                R = self.solve_procrustes(Fhat, F)
                rotated_Fhat = Fhat@R             
                rotated_fx[:,i:i+j] = rotated_Fhat

                i = i + j

            fig, ax = plt.subplots(k,1,figsize=(10,5*k))
            for i in range(k):
                ax[i].plot(t,eigfuncs[:,i],lw=2,color='black',label='true')
                ax[i].plot(t,rotated_fx[:,i],lw=2,color='blue',label='pred')
                ax[i].legend()
                ax[i].set_title(f'True eigval: {eigvals[i]}. Predicted eigval: {solver.eigvals[i]:.3f}')
        
            return fig, ax
        elif solver.dim == 2:
            tmin, tmax = np.quantile(x[:,0], [0.1,0.9])
            tx = np.linspace(tmin, tmax,100)[:,None]
            tmin, tmax = np.quantile(x[:,1], [0.1,0.9])
            ty = np.linspace(tmin, tmax,100)[:,None]

            tx, ty = np.meshgrid(tx, ty)
            # Reshape the grid into an Nx2 array for the function
            grid = np.stack([tx.ravel(), ty.ravel()], axis=-1)

            fx = solver.predict(grid)[:,:k]
            i = 0
            eigvals = self.energy.exact_eigvals(k)
            eigfuncs =  self.energy.exact_eigfunctions(grid, k)
            rotated_fx = np.zeros_like(fx)
            while i < k:
                cur_eigval = eigvals[i]
                e = 1
                j = 1
                while i + j < k and eigvals[i+j] == cur_eigval:
                    e += 1
                    j += 1
                # i = first index of eigenspace
                # i + j - 1 = last index of eigenspace
                # e = dimension of eigenspace
                F = eigfuncs[:,i:i+j]
                Fhat = fx[:,i:i+j]

                R = self.solve_procrustes(Fhat, F)
                rotated_Fhat = Fhat@R
                rotated_fx[:,i:i+j] = rotated_Fhat

                i = i + j
            
            fig, ax = plt.subplots(k,2,figsize=(10,5*k),sharey=True)
            for i in range(k):
                zhat = rotated_fx[:,i].reshape(tx.shape)
                z = eigfuncs[:,i].reshape(tx.shape)
                # Determine the global min and max for consistent color mapping
                z_min = min(z.min(), zhat.min())
                z_max = max(z.max(), zhat.max())

                # Plot first contour
                contour1 = ax[i,0].contourf(tx, ty, zhat, levels=np.linspace(z_min, z_max, 10), cmap='viridis')
                ax[i,0].set_title(f'Predicted eigval: {solver.eigvals[i]:.3f}')

                # Plot second contour
                contour2 = ax[i,1].contourf(tx, ty, z, levels=np.linspace(z_min, z_max, 10), cmap='viridis')
                ax[i,1].set_title(f'True eigval: {eigvals[i]}')


                # Add a shared colorbar
                fig.colorbar(contour1, ax=ax[i,:], orientation='vertical', label='Function Value')

            return fig, ax
    @staticmethod
    def solve_procrustes(A, B):
        """
        Solve the orthogonal Procrustes problem for inputs of shape (n, d).

        Given two matrices A and B each of shape (n, d), this finds the orthogonal
        matrix R (d x d) that minimizes the Frobenius norm || A R - B ||_F.

        Parameters
        ----------
        A : ndarray of shape (n, d)
        B : ndarray of shape (n, d)

        Returns
        -------
        R : ndarray of shape (d, d)
            Orthogonal matrix (R^T R = I) that best maps A onto B.
            i.e.  A @ R ≈ B.
        """
        # Cross-covariance matrix
        M = A.T @ B  # shape (d, d)

        # SVD of M
        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # Orthogonal matrix that aligns A to B
        R = U @ Vt

        return R
        
        
        


        