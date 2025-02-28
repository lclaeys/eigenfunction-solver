import torch

def compute_loss(model, x, Rx, beta):
    """
    Compute variational loss given samples x \sim \mu, Rx and beta
    Args:
        model (nn.Module): model
        x (torch.tensor): samples
        Rx (torch.tensor):  R(samples)
        beta (float): beta value for regularization
    Returns:
        loss (tensor)
    """
    fx = model(x)
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

    return var_loss + 1/beta*orth_loss, var_loss, orth_loss

def compute_l2_error(target_control,
                     learned_control):
        
    norm_sqd_diff = torch.sum(
            (target_control - learned_control) ** 2
            / (target_control.shape[0] * target_control.shape[1])
        )
    
    return norm_sqd_diff
