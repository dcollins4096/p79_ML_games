import torch

class powerspectrum:
    def __init__(self, arr: torch.Tensor):
        """
        arr: Tensor of shape [B, C, N, M] (2D) or [B, C, N, N, N] (3D)
        """
        assert arr.ndim in (4, 5), "Input must be [B,C,N,M] or [B,C,N,N,N]"
        device = arr.device
        rank = arr.ndim - 2  # 2 for 2D, 3 for 3D

        arr = arr.to(torch.float32)

        # FFT over spatial dimensions only
        Nhat = torch.fft.fftn(arr, dim=tuple(range(-rank, 0)))
        Nhat /= arr[..., 0, 0].numel()  # normalize by number of points
        rhohat = Nhat.abs() ** 2  # [B,C,N,M]

        nz = rhohat.shape[-1]
        kx = torch.fft.fftfreq(nz, device=device) * nz  # <<< device-aware
        self.dk = kx[1] - kx[0]
        self.dx = 1 / arr.shape[-1]

        kabs = torch.sort(torch.unique(kx.abs())).values

        # Construct k-grid on same device
        if rank == 2:
            kkx, kky = torch.meshgrid(kx, kx, indexing='ij')
            k = torch.sqrt(kkx**2 + kky**2)
        else:  # rank == 3
            kkx, kky, kkz = torch.meshgrid(kx, kx, kx, indexing='ij')
            k = torch.sqrt(kkx**2 + kky**2 + kkz**2)

        bins = torch.cat([kabs, kabs[-1:] + self.dk]).to(device)

        # Bin indices must be on same device
        bin_indices = torch.bucketize(k.flatten(), bins, right=False) - 1
        bin_indices = bin_indices.clamp(min=0, max=len(bins) - 2)

        # Flatten spatial dims for binning
        rho_flat = rhohat.reshape(rhohat.shape[0], rhohat.shape[1], -1)

        power_per_bin = torch.zeros(
            (arr.shape[0], arr.shape[1], len(bins) - 1),
            dtype=rhohat.dtype,
            device=device
        )

        # Expand bin_indices for broadcasting (and ensure correct device)
        bin_indices_expanded = bin_indices.to(device).expand(rho_flat.shape[0], rho_flat.shape[1], -1)

        # Scatter-add along last dimension
        power_per_bin.scatter_add_(
            dim=2,
            index=bin_indices_expanded,
            src=rho_flat
        )

        bc = 0.5 * (bins[1:] + bins[:-1])
        self.Nhat = Nhat
        self.rho = arr
        self.rhohat = rhohat
        self.k = k
        self.da = self.dk ** rank
        self.power = power_per_bin.real * self.da
        self.kcen = bc

        if rank == 2:
            volume = 2 * torch.pi * self.kcen
        else:
            volume = 4 * torch.pi * self.kcen**2

        # Normalize by number of samples per bin
        counts = torch.bincount(bin_indices, minlength=len(bins)-1).clamp(min=1).to(device)
        volume = volume * counts
        self.avgpower = self.power / volume

