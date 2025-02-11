import torch
import torch.nn as nn

class FourierBasis:
    """
    Basis from sine and cosine functions with given frequency vectors.

    Args:
        frequencies (torch.Tensor): A tensor of shape (F, d) specifying the
            frequency vectors. F = number of frequencies, d = dimension of x.
    
    Example:
        frequencies = torch.tensor([[1.0, 0.0], 
                                    [0.0, 2.0]])  # 2 frequencies in 2D
        basis = FourierBasis(frequencies)  
        
        x = torch.randn(5, 2)    # 5 query points in 2D
        Bx = basis(x)            # [5, 4] => cos(k1·x), sin(k1·x), cos(k2·x), sin(k2·x)
        dBx = basis.grad(x)      # [5, 2, 4]
        lapBx = basis.laplacian(x)  # [5, 4]
    """
    def __init__(self, frequencies: torch.Tensor):
        """
        frequencies: shape (F, d) 
            - F: number of frequency vectors
            - d: dimension of the input space
        """
        self.frequencies = frequencies  # (F, d)
        self.dim = frequencies.shape[1]
        self.num_freqs = frequencies.shape[0]
        # We get 2 basis functions (cos, sin) per frequency vector
        self.basis_dim = 2 * self.num_freqs

        # Precompute squared norms of each frequency vector for the Laplacian
        with torch.no_grad():
            self.freq_norms = (frequencies ** 2).sum(dim=1)  # shape (F,)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all basis functions at points x.

        Args:
            x (torch.Tensor): shape (N, d), where N is # of points, d is dimension.

        Returns:
            torch.Tensor of shape (N, 2*F):
                [cos(k1·x), sin(k1·x), ..., cos(kF·x), sin(kF·x)]
        """
        # Ensure frequencies are on the same device/dtype as x
        if x.dtype != self.frequencies.dtype:
            self.frequencies = self.frequencies.to(x.dtype)
            self.freq_norms = self.freq_norms.to(x.dtype)

        # Dot product x·k => shape (N, F)
        dot_products = x @ self.frequencies.T  # (N, F)

        # cos and sin
        cos_part = torch.cos(dot_products)  # (N, F)
        sin_part = torch.sin(dot_products)  # (N, F)

        return torch.cat([cos_part, sin_part], dim=-1)  # (N, 2F)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient d/dx of each basis function at points x.

        For each frequency k:
            grad_x cos(k·x) = -k * sin(k·x)
            grad_x sin(k·x) =  k * cos(k·x)

        Args:
            x (torch.Tensor): shape (N, d)

        Returns:
            torch.Tensor of shape (N, 2*F, d), i.e. the gradient w.r.t. each dimension.
        """
        # Match device/dtype
        if x.dtype != self.frequencies.dtype:
            self.frequencies = self.frequencies.to(x.dtype)
            self.freq_norms = self.freq_norms.to(x.dtype)

        dot_products = x @ self.frequencies.T  # (N, F)
        cos_part = torch.cos(dot_products)     # (N, F)
        sin_part = torch.sin(dot_products)     # (N, F)

        # We want shape (N, d, F) for broadcasting
        freq_t = self.frequencies.T.unsqueeze(0)  # (1, d, F)

        # grad of cos(k·x) = -k sin(k·x)
        grad_cos = -freq_t * sin_part.unsqueeze(1)  # (N, d, F)

        # grad of sin(k·x) =  k cos(k·x)
        grad_sin = freq_t * cos_part.unsqueeze(1)   # (N, d, F)


        # Combine [cos-basis, sin-basis] along last dimension => (N, d, 2F)
        return torch.cat([grad_cos, grad_sin], dim=-1).transpose(1,2)

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Laplacian (sum of 2nd partials) of each basis function at points x.

        For each frequency k:
            Δ cos(k·x) = -|k|^2 cos(k·x)
            Δ sin(k·x) = -|k|^2 sin(k·x)

        Args:
            x (torch.Tensor): shape (N, d)

        Returns:
            torch.Tensor of shape (N, 2*F)
        """
        # Match device/dtype
        if x.dtype != self.frequencies.dtype:
            self.frequencies = self.frequencies.to(x.dtype)
            self.freq_norms = self.freq_norms.to(x.dtype)

        dot_products = x @ self.frequencies.T  # (N, F)
        cos_part = torch.cos(dot_products)     # (N, F)
        sin_part = torch.sin(dot_products)     # (N, F)

        # Multiply each basis by -|k|^2
        factor = -self.freq_norms.unsqueeze(0)  # shape (1, F)
        lap_cos = factor * cos_part            # (N, F)
        lap_sin = factor * sin_part            # (N, F)

        return torch.cat([lap_cos, lap_sin], dim=-1)  # (N, 2F)
