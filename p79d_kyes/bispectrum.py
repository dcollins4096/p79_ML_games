import torch

def make_kgrid(N, M, device):
    """Fourier-space wavevector magnitudes for 2D FFT grid."""
    ky = torch.fft.fftfreq(N, d=1.0).to(device) * N
    kx = torch.fft.fftfreq(M, d=1.0).to(device) * M
    KX, KY = torch.meshgrid(kx, ky, indexing="xy")
    return torch.sqrt(KX**2 + KY**2)  # [N, M]


def make_bin_masks(k_mag, n_bins=10, kmax=None):
    """Precompute boolean masks for Fourier bins."""
    if kmax is None:
        kmax = k_mag.max().item()
    edges = torch.linspace(0, kmax, n_bins + 1, device=k_mag.device)
    bin_idx = torch.bucketize(k_mag.flatten(), edges) - 1
    bin_idx = bin_idx.reshape_as(k_mag)

    masks = []
    for i in range(n_bins):
        mask = (bin_idx == i)
        if mask.any():
            masks.append(mask.float())
        else:
            masks.append(torch.zeros_like(k_mag))
    return torch.stack(masks, dim=0)  # [n_bins, N, M]


def compute_bispectrum1(y, bin_masks):
    """
    Compute a binned bispectrum estimate for a batch of fields.
    Args:
      y: [B,C,N,M] input fields (real space)
      bin_masks: [n_bins, N, M] Fourier-space masks
    Returns:
      bispec: [B,C,n_bins,n_bins] bispectrum values
    """
    B, C, N, M = y.shape
    device = y.device
    n_bins = bin_masks.shape[0]

    # Fourier transforms
    Y = torch.fft.fftn(y, dim=(-2, -1))

    # Apply masks in Fourier space: [B,C,n_bins,N,M]
    masked = Y.unsqueeze(2) * bin_masks[None, None, :, :, :]

    # Back to real space (convolution representation)
    conv = torch.fft.ifftn(masked, dim=(-2, -1)).real  # [B,C,n_bins,N,M]

    # Contract: < conv(bin_i) * field(bin_j) >
    # Flatten spatial dim for efficient einsum
    conv = conv.reshape(B, C, n_bins, -1)  # [B,C,n_bins,N*M]
    y_flat = y.reshape(B, C, -1)           # [B,C,N*M]

    # Compute bispectrum: [B,C,n_bins,n_bins]
    bispec = torch.einsum("bcin,bcm->bcim", conv, y_flat) / (N*M)

    return bispec


def bispectrum_loss(y_pred, y_true, bin_masks):
    """
    Compare bispectra of prediction and truth with MSE.
    """
    bispec_pred = compute_bispectrum(y_pred, bin_masks)
    bispec_true = compute_bispectrum(y_true, bin_masks)
    return torch.mean((bispec_pred - bispec_true) ** 2)

import torch
import torch.fft as fft

def compute_bispectrum2(x):
    """
    Compute exact 2D bispectrum keeping output [B, C, N, M], fully vectorized.
    
    Args:
        x: Tensor of shape [B, C, N, M]
    
    Returns:
        bispec: Tensor of shape [B, C, N, M] (complex)
    """
    B, C, N, M = x.shape
    X = fft.fft2(x)  # [B, C, N, M]

    # Create 2D index grids for last two dimensions
    ky = torch.arange(N, device=x.device).view(N, 1).expand(N, M)  # [N, M]
    kx = torch.arange(M, device=x.device).view(1, M).expand(N, M)  # [N, M]

    # Compute 2k indices modulo N and M
    ky2 = (2 * ky) % N  # [N, M]
    kx2 = (2 * kx) % M  # [N, M]

    # Expand indices to match batch and channel dimensions
    ky2 = ky2.unsqueeze(0).unsqueeze(0).expand(B, C, N, M)  # [B,C,N,M]
    kx2 = kx2.unsqueeze(0).unsqueeze(0).expand(B, C, N, M)  # [B,C,N,M]

    # Create flattened indices for gather
    batch_idx = torch.arange(B, device=x.device)[:, None, None, None].expand(B, C, N, M)
    channel_idx = torch.arange(C, device=x.device)[None, :, None, None].expand(B, C, N, M)

    # Gather F*(2k) using advanced indexing
    X2 = X[batch_idx, channel_idx, ky2, kx2]  # [B, C, N, M]

    # Compute bispectrum
    bispec = X * X * torch.conj(X2)

    return bispec
import torch
import torch.fft as fft

def compute_bispectrum_torch(data, nsamples=100, mean_subtract=False, seed=1000, device='cuda'):
    """
    Compute the bispectrum in a Monte Carlo sampling fashion, vectorized over [B,C,N,M].
    
    Args:
        data: Tensor of shape [B, C, N, M]
        nsamples: Number of samples per magnitude pair
        mean_subtract: Subtract mean before FFT
        seed: RNG seed
        device: 'cuda' or 'cpu'
        
    Returns:
        bispectrum: [B, C, N//2, M//2] complex tensor
        bicoherence: [B, C, N//2, M//2] real tensor
    """
    torch.manual_seed(seed)
    
    B, C, N, M = data.shape
    
    if mean_subtract:
        data = data - data.mean(dim=(-2,-1), keepdim=True)
    
    # 2D FFT
    X = fft.fft2(data)  # [B,C,N,M]
    X_conj = torch.conj(X)
    
    # Prepare bispectrum accumulator
    bispec_shape = (N//2, M//2)
    bispectrum = torch.zeros((B, C, *bispec_shape), dtype=torch.complex64, device=device)
    bicoherence = torch.zeros((B, C, *bispec_shape), dtype=torch.float32, device=device)
    
    # Generate random angles for Monte Carlo sampling
    phi1 = 2 * torch.pi * torch.rand((nsamples,), device=device)
    phi2 = 2 * torch.pi * torch.rand((nsamples,), device=device)
    
    # Generate magnitude grid
    k1_vals = torch.arange(bispec_shape[0], device=device)
    k2_vals = torch.arange(bispec_shape[1], device=device)
    K1, K2 = torch.meshgrid(k1_vals, k2_vals, indexing='ij')  # [N//2, M//2]
    
    # Expand for sampling
    K1 = K1.unsqueeze(-1)  # [N//2, M//2, 1]
    K2 = K2.unsqueeze(-1)  # [N//2, M//2, 1]
    
    # Compute sampled coordinates
    k1x = (K1 * torch.cos(phi1)).long() % N  # [N//2, M//2, nsamples]
    k1y = (K1 * torch.sin(phi1)).long() % M
    k2x = (K2 * torch.cos(phi2)).long() % N
    k2y = (K2 * torch.sin(phi2)).long() % M
    
    # Compute k3 = k1 + k2
    k3x = (k1x + k2x) % N
    k3y = (k1y + k2y) % M
    
    # Vectorized gather from FFT
    # Flatten indices for gather
    B_idx = torch.arange(B, device=device)[:,None,None,None,None]
    C_idx = torch.arange(C, device=device)[None,:,None,None,None]
    
    # Expand to broadcast shapes: [B,C,N//2,M//2,nsamples]
    B_idx = B_idx.expand(B,C,*bispec_shape,nsamples)
    C_idx = C_idx.expand(B,C,*bispec_shape,nsamples)
    
    # Gather FFT values
    X1 = X[B_idx, C_idx, k1x.unsqueeze(0).expand(B,C,*k1x.shape),
                 k1y.unsqueeze(0).expand(B,C,*k1y.shape)]
    X2 = X[B_idx, C_idx, k2x.unsqueeze(0).expand(B,C,*k2x.shape),
                 k2y.unsqueeze(0).expand(B,C,*k2y.shape)]
    X3 = X_conj[B_idx, C_idx, k3x.unsqueeze(0).expand(B,C,*k3x.shape),
                         k3y.unsqueeze(0).expand(B,C,*k3y.shape)]
    
    # Compute bispectrum samples
    samps = X1 * X2 * X3  # [B,C,N//2,M//2,nsamples]
    
    # Sum over samples
    bispectrum = samps.sum(dim=-1)
    bicoherence = samps.abs().sum(dim=-1)
    
    return bispectrum, bicoherence

