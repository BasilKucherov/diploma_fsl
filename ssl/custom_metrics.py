import torch
import torch.nn.functional as F

def alignment(z1, z2):
    """
    Alignment metric (Wang & Isola).
    Expected input: L2-normalized embeddings z1, z2 of shape (N, d).
    align = E[|z - z+|^2]
    """
    return (z1 - z2).norm(p=2, dim=1).pow(2).mean()

def uniformity(z, t=2):
    """
    Uniformity metric (Wang & Isola).
    Expected input: L2-normalized embeddings z of shape (N, d).
    unif = log E[exp(-t * |z_i - z_j|^2)]
    """
    N = z.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=z.device)
    
    # pdist matrix: |x-y|^2 = |x|^2 + |y|^2 - 2<x,y> = 2 - 2<x,y> (since normed)
    # explicit computation:
    # dist_sq[i, j] = ||z[i] - z[j]||^2
    dist_sq = torch.cdist(z, z, p=2).pow(2)
    
    # We want sum over i != j. Mask diagonal.
    mask = torch.eye(N, dtype=torch.bool, device=z.device)
    dist_sq_masked = dist_sq[~mask].view(N, N - 1)
    
    return torch.log(torch.exp(-t * dist_sq_masked).mean())

def vicreg_invariance(z1, z2):
    """
    Invariance term from VICReg.
    Mean squared distance between unnormalized embeddings.
    """
    return F.mse_loss(z1, z2)

def vicreg_variance(z, gamma=1.0, epsilon=1e-4):
    """
    Variance term from VICReg.
    Hinge loss on standard deviation of each dimension.
    v = 1/d * sum_j max(0, gamma - std(z_j))
    """
    # z: (N, d)
    std = torch.sqrt(z.var(dim=0) + epsilon)
    return torch.nn.functional.relu(gamma - std).mean()

def vicreg_covariance(z):
    """
    Covariance term from VICReg.
    Off-diagonal sum of squared covariance.
    """
    N, D = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (N - 1)
    
    # off-diagonal elements
    off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
    return cov[off_diag_mask].pow(2).sum() / D

