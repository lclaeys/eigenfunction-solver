import numpy as np

def eigen_error(x,solver,k):
    # MAD |f(x) - Lf(x)|, average over first k eigenfunctions
    fx = solver.predict(x)[:,:k]
    Lfx = solver.predict_Lf(x)[:,:k]

    err = np.median(np.abs(fx-Lfx).sum(axis=0)).mean()

    return err

def orth_error(x,solver,k):
    # MSE E[f(x)f(x)^T - I]
    fx = solver.predict(x)[:,:k]

    cov = fx.T@fx/x.shape[0]
    err = np.mean((cov - np.eye(cov.shape[0]))**2)
    return err

def L_prod_error(x,solver,k):
    grad_fx = solver.predict_grad(x)

    cost = np.diag(np.sum(np.matmul(grad_fx,np.transpose(grad_fx,axes=[0,2,1])),axis=0))
    return np.sum(cost[:k])
    
