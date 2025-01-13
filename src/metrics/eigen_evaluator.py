import numpy as np
import matplotlib.pyplot as plt

class EigenEvaluator():
    """
    Evaluator class. The idea is to be able to pass a solver object and an energy function, and evaluate the quality of the learned eigenfunctions through various metrics.
    Also includes some methods for plotting results.
    """
    def __init__(self, energy):
        self.energy = energy
        self.exact_eigfuncs = None
        self.rng = np.random.default_rng(42)

    def evaluate_metrics(self, solver, x, metrics = None, k = 1):
        """
        Compute evaluation metrics

        Args:
            solver (BaseSolver): fitted solver object
            x (ndarray)[N,d]: points used for evaluation (for example, computing MSE between fitted and true eigenfunction)
            metrics (array): array of metrics to evaluate
            k (int): evaluate up to k-th eigenfunction/eigenvalue
        Returns:
            out (dict): keys are the metrics, values are k-dimensional arrays with the error computed for the first k eigenfunctions.
        """
        out = {}
        fx = None
        Lfx = None
        grad_fx = None
        fitted_eigvals = None
        samples = solver.samples

        for metric in metrics:
            if metric == "eigen_error":
                """
                MSE of Lf(x) - lambda*f(x)
                """
                if fx is None:
                    fx = solver.predict(x)[:,:k]
                if Lfx is None:
                    Lfx = solver.predict_Lf(x)[:,:k]
                errs = np.mean((solver.eigvals[:k]*fx-Lfx)**2,axis=0)
                out[metric] = np.cumsum(errs)/np.arange(1,k+1)

            if metric == "fitted_eigen_error":
                # TODO: what is this for?
                """
                MSE of Lf(x) - lambda*f(x), computed using fitted eigenvalues.
                """
                if fitted_eigvals is None:
                    fitted_eigvals = solver.fit_eigvals(x)
                if fx is None:
                    fx = solver.predict(x)[:,:k]
                if Lfx is None:
                    Lfx = solver.predict_Lf(x)[:,:k]
                errs = np.mean((fitted_eigvals[:k]*fx-Lfx)**2,axis=0)
                out[metric] = np.cumsum(errs)/np.arange(1,k+1)


            if metric == "orth_error":
                """
                \|f(x)f(x)^T -  diag(f(x)f(x)^T)\|^2
                """
                if fx is None:
                    fx = solver.predict(x)[:,:k]
                cov = fx.T@fx/x.shape[0]
                cov_err = cov - np.eye(cov.shape[0])
                errs = np.array([np.mean(cov_err[:i,:i]**2) for i in range(1,k+1)])
                out[metric] = errs

            if metric == "eigen_cost":
                """
                sum_i^k <grad f_i , grad f_i >
                """                
                if grad_fx is None:
                    x_mu = self.rng.choice(samples,solver.num_samples,axis=0)
                    grad_fx = solver.predict_grad(x_mu)[:,:k,:]
                
                costs = np.diag(np.mean(np.matmul(grad_fx,np.transpose(grad_fx,axes=[0,2,1])),axis=0))
                out[metric] = np.cumsum(costs)

            if metric == "eigenvalue_mse":
                """
                MSE of eigenvalues
                """
                eigvals = self.energy.exact_eigvals(k)
                errs = (eigvals[:k]-np.array(solver.eigvals)[:k])**2
                out[metric] = np.cumsum(errs)/np.arange(1,k+1)

            if metric == "fitted_eigenvalue_mse":
                """
                MSE of fitted eigenvalues
                """
                if fitted_eigvals is None:
                    fitted_eigvals = solver.fit_eigvals(x)
                errs = (eigvals[:k]-fitted_eigvals[:k])**2
                out[metric] = np.cumsum(errs)/np.arange(1,k+1)

            if metric == "eigenfunc_mse":
                """
                MSE of eigenfunction. Finds the rotation in each eigenspace that minimizes MSE.
                """

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
                out[metric] = np.cumsum(errs)/np.arange(1,k+1)
                
        return out
    
    def plot(self, solver, x, k, plot_exact = False, plot_Lf = False):
        """
        Plot the first k eigenfunctions computed by the given solver at a grid defined using 5 and 95$ quantiles of x in each dimension.

        Only works for d=1 or d=2.
        
        Args:
            solver (BaseSolver): fitted solver object
            x (ndarray): points
            k (int): number of eigfuncs to plot
            plot_exact (bool): plot exact eigenfunctions (should be supported by energy)
            plot_Lf (bool): plot Lf/lambda of fitted eigenfunctions (should be supported by solver)

        Returns:
            fig, ax: figure objects
        """

        if solver.dim == 1:
            tmin, tmax = np.quantile(x, [0.05,0.95])
            t = np.linspace(tmin, tmax,1000)[:,None]
            
            fx = solver.predict(t)[:,:k]
            
            if plot_Lf:
                Lfx = solver.predict_Lf(t)[:,:k]

            i = 0
            

            rotated_fx = np.zeros_like(fx)
            
            if plot_Lf:
                rotated_Lfx = np.zeros_like(Lfx)
                fitted_eigvals = solver.fit_eigvals(x)

            # rotate eigenfunctions to match exact solution
            if plot_exact:
                eigvals = self.energy.exact_eigvals(k)
                eigfuncs =  self.energy.exact_eigfunctions(t, k)
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
                    if plot_Lf:
                        rotated_Lfx[:,i:i+j] = Lfx[:,i:i+j]@R

                    i = i + j

                fx = rotated_fx
                if plot_Lf:
                    Lfx = rotated_Lfx

            fig, ax = plt.subplots(k,1,figsize=(10,5*k))
            
            for i in range(k):
                title = ""
                if plot_exact:
                    ax[i].plot(t,eigfuncs[:,i],lw=2,color='black',label='true')
                    title += f"True eigval: {eigvals[i]:.3f}. "
                
                if plot_Lf:
                    ax[i].plot(t,Lfx[:,i]/fitted_eigvals[i],lw=2,color='red',label='Lfx/lambda')
                    title += f"Fitted eigval: {fitted_eigvals[i]:.3f}. "
                
                ax[i].plot(t,fx[:,i],lw=2,color='blue',label='pred')
                title += f"Method eigval: {solver.eigvals[i]:.3f}."
                
                ax[i].legend()
                ax[i].set_title(title)
        
            return fig, ax
        
        elif solver.dim == 2:
            tmin, tmax = np.quantile(x[:,0], [0.05,0.95])
            tx = np.linspace(tmin, tmax,100)[:,None]
            tmin, tmax = np.quantile(x[:,1], [0.05,0.95])
            ty = np.linspace(tmin, tmax,100)[:,None]

            tx, ty = np.meshgrid(tx, ty)
            # Reshape the grid into an Nx2 array for the function
            grid = np.stack([tx.ravel(), ty.ravel()], axis=-1)

            fx = solver.predict(grid)[:,:k]

            if plot_Lf:
                Lfx = solver.predict_Lf(grid)[:,:k]
                fitted_eigvals = solver.fit_eigvals(x)
                rotated_Lfx = np.zeros_like(Lfx)

            if plot_exact:
                eigvals = self.energy.exact_eigvals(k)
                eigfuncs =  self.energy.exact_eigfunctions(grid, k)
                i = 0

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
                    if plot_Lf:
                        rotated_Lfx[:,i:i+j] = Lfx[:,i:i+j]@R

                    i = i + j
                fx = rotated_fx
                if plot_Lf:
                    Lfx = rotated_Lfx
            
            num_plots = 1 + int(plot_Lf) + int(plot_exact)
            fig, ax = plt.subplots(k,num_plots,figsize=(num_plots*4,5*k),sharey=True)
            
            for i in range(k):
                zvalues = []
                titles = []

                zvalues.append(fx[:,i].reshape(tx.shape))
                titles.append(f'Predicted eigval: {solver.eigvals[i]:.3f}')

                if plot_Lf:
                    zvalues.append(Lfx[:,i].reshape(tx.shape)/fitted_eigvals[i])
                    titles.append(f'Fitted eigval: {fitted_eigvals[i]:.3f}')
                
                if plot_exact:
                    zvalues.append(eigfuncs[:,i].reshape(tx.shape))
                    titles.append(f'True eigval: {eigvals[i]}')
                

                # Determine the global min and max for consistent color mapping
                z_min = min([z.min() for z in zvalues])
                z_max = max([z.max() for z in zvalues])
                
                # avoid error if functions are constant
                if z_min == z_max:
                    z_min -= 1e-6
                    z_max += 1e-6

                if num_plots > 1:
                    for j in range(num_plots):
                        contour = ax[i,j].contourf(tx, ty, zvalues[j], levels=np.linspace(z_min, z_max, 10), cmap='viridis')
                        ax[i,j].set_title(titles[j])
                    fig.colorbar(contour, ax=ax[i,:], orientation='vertical', label='Function Value')

                else:
                    contour = ax[i].contourf(tx, ty, zvalues[0], levels=np.linspace(z_min, z_max, 10), cmap='viridis')
                    ax[i].set_title(titles[0])
                    fig.colorbar(contour, ax=ax[i], orientation='vertical', label='Function Value')

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
        
        
        


        