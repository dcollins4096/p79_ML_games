import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
import time
import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import loader
import os
from torch.nn.utils import spectral_norm
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import pdb

idd = 1002
what = "GAN 1000 with net0184 for the generator"

dirpath = "/home/dcollins/repos/p79_ML_games/p79d_kyes/datasets/"
#dirpath = "./"

fname_train = "p79d_subsets_S512_N3_xyz__down_128Athena_TQU_first.h5"
fname_valid = "p79d_subsets_S512_N3_xyz__down_128Athena_TQU_second.h5"


ntrain = 14000
nvalid = 100
ntest = 5000
downsample = 128
img_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Hyperparameters
epochs = 500
steps = 1000
lr = 1e-4
batch_size = 64
lr_schedule = [1000]
weight_decay = 0.01

# Hybrid architecture parameters
cnn_output_size = 64  # After CNN pooling: 128 → 64
patch_size = 8  # Smaller patches on reduced resolution
embed_dim = 384
depth = 6
num_heads = 6
mlp_ratio = 4.0
dropout = 0.1


def load_data():
    print('read the data')
    train_raw = loader.loader(dirpath + fname_train, ntrain=ntrain, nvalid=0)
    valid_raw = loader.loader(dirpath + fname_valid, ntrain=1, nvalid=nvalid)
    
    all_data_combined = train_raw['train']
    all_ms_combined = train_raw['quantities']['train']['Ms_act']
    
    all_test_data = valid_raw['test']
    all_test_ms = valid_raw['quantities']['test']['Ms_act']
    
    # Stratified validation
    mach_bins = [0, 4, 6, 8, 10, 15]
    samples_per_bin = nvalid // (len(mach_bins) - 1)
    
    valid_indices = []
    for i in range(len(mach_bins) - 1):
        mask = (all_ms_combined >= mach_bins[i]) & (all_ms_combined < mach_bins[i+1])
        bin_indices = torch.where(torch.from_numpy(mask))[0].cpu()
        
        if len(bin_indices) >= samples_per_bin:
            selected = bin_indices[torch.randperm(len(bin_indices))[:samples_per_bin]]
            valid_indices.extend(selected.tolist())
        else:
            valid_indices.extend(bin_indices.tolist())
            print(f"Warning: Only {len(bin_indices)} samples in Ms range [{mach_bins[i]}, {mach_bins[i+1]})")
    
    valid_indices = torch.tensor(valid_indices)
    
    valid_data = all_data_combined[valid_indices]
    valid_ms = all_ms_combined[valid_indices]
    
    all_indices = torch.arange(len(all_data_combined))
    train_mask = ~torch.isin(all_indices, valid_indices)
    train_data = all_data_combined[train_mask]
    train_ms = all_ms_combined[train_mask]
    
    all_data = {
        'train': train_data,
        'valid': valid_data,
        'test': all_test_data[:ntest],
        'quantities': {}
    }
    
    all_data['quantities']['train'] = {
        'Ms_act': train_ms,
        'Ma_act': train_raw['quantities']['train']['Ma_act'][train_mask]
    }
    all_data['quantities']['valid'] = {
        'Ms_act': valid_ms,
        'Ma_act': train_raw['quantities']['train']['Ma_act'][valid_indices]
    }
    all_data['quantities']['test'] = {
        'Ms_act': all_test_ms[:ntest],
        'Ma_act': train_raw['quantities']['train']['Ma_act'][:ntest]
    }
    
    print(f'Train: {len(train_data)}, Valid (stratified): {len(valid_data)}, Test: {min(ntest, len(all_test_data))}')
    
    print('\nValidation set Mach distribution:')
    for i in range(len(mach_bins)-1):
        mask = (valid_ms >= mach_bins[i]) & (valid_ms < mach_bins[i+1])
        print(f"  Ms [{mach_bins[i]:2d}-{mach_bins[i+1]:2d}): {mask.sum():3d} samples")
    
    print('done')
    return all_data


def thisnet():
    model = HybridCNNViT(
        img_size=img_size,
        cnn_output_size=cnn_output_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=1,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout
    )
    model = model.to(device)
    return model

@torch.no_grad()
def estimate_log_stats(x, n=2048, eps=0.0):
    # x: [N, C, H, W] torch tensor or numpy -> torch
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    x = x[:min(n, x.shape[0])].float()
    x = x[:, 0:1]  # density channel
    x = torch.log1p(torch.clamp_min(x, 0.0))  # safe log
    mean = x.mean().item()
    std  = x.std().item()
    return mean, std

def downsample_avg(x, M):
    if x.ndim == 2:   # [N, N]
        x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, N, N]
        out = F.adaptive_avg_pool2d(x, (M, M))
        return out.squeeze(0).squeeze(0) # -> [M, M]
    elif x.ndim == 4: # [B, C, N, N]
        return F.adaptive_avg_pool2d(x, (M, M))
    else:
        raise ValueError("Input must be [N, N] or [B, C, N, N]")


class TurbulenceDataset(Dataset):
    def __init__(self, all_data, augment=False, mean=0.0, std=1.0,
                 in_idx=0, out_idx=(1,2)):
        self.all_data = all_data
        if downsample:
            self.all_data = downsample_avg(all_data, downsample)
        self.augment = augment
        self.mean = float(mean)
        self.std  = float(std)
        self.in_idx = in_idx
        self.out_idx = out_idx

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        x = self.all_data[idx][self.in_idx:self.in_idx+1].float()     # [1,H,W]
        y = self.all_data[idx][list(self.out_idx)].float()            # [2,H,W]

        # same transform for x (and often for y, if it’s same kind of field)
        # If y is velocity with negatives, DO NOT log1p clamp like density.
        x = torch.log1p(torch.clamp_min(x, 0.0))
        x = (x - self.mean) / (self.std + 1e-6)
        x = x.clamp(-5, 5)

        # optional: standardize y too, but usually with its own mean/std (recommended).
        # For now, leave y as-is or z-score with precomputed stats.

        if self.augment:
            H, W = x.shape[-2:]
            dy = torch.randint(0, H, (1,)).item()
            dx = torch.randint(0, W, (1,)).item()
            x = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
            y = torch.roll(y, shifts=(dy, dx), dims=(-2, -1))
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[-1]); y = torch.flip(y, dims=[-1])
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[-2]); y = torch.flip(y, dims=[-2])

        return x, y



# ----------------------------
# Utilities: conditioning
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True):
        super().__init__()
        if down:
            self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        else:
            self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.silu(self.norm(self.conv(x)))

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base=64):
        super().__init__()
        # encoder
        self.e1 = nn.Conv2d(in_ch, base, 4, 2, 1)          # 128->64 (no BN)
        self.e2 = ConvBlock(base, base*2, down=True)       # 64->32
        self.e3 = ConvBlock(base*2, base*4, down=True)     # 32->16
        self.e4 = ConvBlock(base*4, base*8, down=True)     # 16->8
        self.e5 = ConvBlock(base*8, base*8, down=True)     # 8->4 bottleneck

        # decoder (skip connections)
        self.d1 = ConvBlock(base*8, base*8, down=False)    # 4->8
        self.d2 = ConvBlock(base*16, base*4, down=False)   # 8->16
        self.d3 = ConvBlock(base*8, base*2, down=False)    # 16->32
        self.d4 = ConvBlock(base*4, base, down=False)      # 32->64
        self.d5 = nn.ConvTranspose2d(base*2, out_ch, 4, 2, 1)  # 64->128

    def forward(self, x, z=None):
        # If you want noise, inject it by concatenation at bottleneck:
        # (z shaped to [B, Z, 4,4]) and concat with e5 output.
        e1 = F.silu(self.e1(x))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b  = self.e5(e4)   # [B, 8*base, 4,4]

        if z is not None:
            # z: [B, z_dim] -> [B, z_dim, 4,4]
            zmap = z.view(z.size(0), z.size(1), 1, 1).expand(-1, -1, b.size(2), b.size(3))
            b = torch.cat([b, zmap], dim=1)

        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e4], dim=1))
        d3 = self.d3(torch.cat([d2, e3], dim=1))
        d4 = self.d4(torch.cat([d3, e2], dim=1))
        y  = self.d5(torch.cat([d4, e1], dim=1))  # [B,2,128,128]
        return y

class ResidualBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, pool_type="avg", dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(p=dropout_p) 

        # Skip connection if channel mismatch
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1)

        # --- Pooling variants ---
        self.pool_type = pool_type
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool_type == "avgmax":
            # Concatenate avg + max → doubles channels for fc1
            self.pool_avg = nn.AdaptiveAvgPool2d(1)
            self.pool_max = nn.AdaptiveMaxPool2d(1)
        elif pool_type == "learned":
            # 1x1 conv to learn pooling weights (H×W → 1)
            self.pool = nn.Conv2d(out_channels, 1, kernel_size=1)

        # --- SE MLP ---
        if pool_type == "avgmax":
            se_in = 2 * out_channels
        else:
            se_in = out_channels

        self.fc1 = nn.Linear(se_in, out_channels // reduction)
        self.fc2 = nn.Linear(out_channels // reduction, out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # --- SE attention pooling ---
        if self.pool_type == "avg":
            w = self.pool(out).view(out.size(0), -1)
        elif self.pool_type == "max":
            w = self.pool(out).view(out.size(0), -1)
        elif self.pool_type == "avgmax":
            w_avg = self.pool_avg(out).view(out.size(0), -1)
            w_max = self.pool_max(out).view(out.size(0), -1)
            w = torch.cat([w_avg, w_max], dim=1)
        elif self.pool_type == "learned":
            # Apply learned 1x1 conv → softmax over spatial dims
            weights = F.softmax(self.pool(out).view(out.size(0), -1), dim=1)
            w = torch.sum(out.view(out.size(0), out.size(1), -1) * weights.unsqueeze(1), dim=-1)

        # --- SE excitation ---
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).view(out.size(0), out.size(1), 1, 1)
        out = out * w

        # Skip connection
        if self.proj is not None:
            identity = self.proj(identity)
        out += identity
        return F.relu(out)

class main_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_channels=32,
                 use_fc_bottleneck=True, fc_hidden=512, fc_spatial=4, rotation_prob=0,
                 use_cross_attention=False, attn_heads=1, epochs=epochs, pool_type='avg', 
                 err_L1=1, err_Multi=1,err_Pear=1,err_SSIM=1,err_Grad=1,err_Power=1,err_Bisp=0,err_Cross=1,
                 suffix='', dropout_1=0, dropout_2=0, dropout_3=0):
        super().__init__()
        arg_dict = locals()
        self.use_fc_bottleneck = use_fc_bottleneck
        self.fc_spatial = fc_spatial
        self.dropout_2=dropout_2
        self.use_cross_attention=use_cross_attention
        self.err_L1=err_L1
        self.err_Multi=err_Multi
        self.err_Pear=err_Pear
        self.err_SSIM=err_SSIM
        self.err_Grad=err_Grad
        self.err_Power=err_Power
        self.err_Bisp=err_Bisp
        self.err_Cross=err_Cross
        self.rotation_prob=rotation_prob
        if 0:
            for arg in arg_dict:
                if arg in ['self','__class__','arg_dict','text','data']:
                    continue
                if type(arg_dict[arg]) == str:
                    text = arg_dict[arg]
                    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)
                else:
                    data = torch.tensor(arg_dict[arg])
                self.register_buffer(arg,data)

        #raise
        #self.use_fc_bottleneck = use_fc_bottleneck
        #self.use_cross_attention = use_cross_attention

        # Encoder
        self.enc1 = ResidualBlockSE(in_channels, base_channels, pool_type=pool_type, dropout_p=dropout_1)
        self.enc2 = ResidualBlockSE(base_channels, base_channels*2, pool_type=pool_type, dropout_p=dropout_1)
        self.enc3 = ResidualBlockSE(base_channels*2, base_channels*4, pool_type=pool_type, dropout_p=dropout_1)
        self.enc4 = ResidualBlockSE(base_channels*4, base_channels*8, pool_type=pool_type, dropout_p=dropout_1)
        self.pool = nn.MaxPool2d(2)

        # Optional FC bottleneck
        if use_fc_bottleneck:
            self.fc1 = nn.Linear(base_channels*8*fc_spatial*fc_spatial, fc_hidden)
            self.fc2 = nn.Linear(fc_hidden, base_channels*8*fc_spatial*fc_spatial)

        # Learned upsampling via ConvTranspose2d
        self.up4 = nn.ConvTranspose2d(base_channels*8, base_channels*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Decoder with skip connections
        self.dec4 = ResidualBlockSE(base_channels*8 + base_channels*4, base_channels*4, pool_type=pool_type, dropout_p=dropout_3)
        self.dec3 = ResidualBlockSE(base_channels*4 + base_channels*2, base_channels*2, pool_type=pool_type, dropout_p=dropout_3)
        self.dec2 = ResidualBlockSE(base_channels*2 + base_channels, base_channels, pool_type=pool_type, dropout_p=dropout_3)
        self.dec1 = nn.Conv2d(base_channels, out_channels, 3, padding=1)

        # --- Multi-scale output heads ---
        self.out_d4 = nn.Conv2d(base_channels*4, out_channels, 3, padding=1)
        self.out_d3 = nn.Conv2d(base_channels*2, out_channels, 3, padding=1)
        self.out_d2 = nn.Conv2d(base_channels,   out_channels, 3, padding=1)
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Optional FC bottleneck
        if self.use_fc_bottleneck:
            B, C, H, W = e4.shape
            z = F.adaptive_avg_pool2d(e4, (self.fc_spatial, self.fc_spatial)).view(B, -1)
            z = F.relu(self.fc1(z))
            z = F.dropout(z, p=self.dropout_2, training=self.training)
            z = F.relu(self.fc2(z))
            z = F.dropout(z, p=self.dropout_2, training=self.training)
            e4 = F.interpolate(z.view(B, C, self.fc_spatial, self.fc_spatial),
                               size=(H, W), mode='bilinear', align_corners=False)

        # Decoder
        d4 = self.up4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        out_main = self.dec1(d2)

        # Multi-scale predictions
        #out_d4 = self.out_d4(d4)
        #out_d3 = self.out_d3(d3)
        #out_d2 = self.out_d2(d2)

        if self.use_cross_attention:
            out_main = self.cross_attn(out_main)

        return out_main #, out_d2, out_d3, out_d4

class FiLM(nn.Module):
    """Feature-wise linear modulation: x * (1+gamma) + beta"""
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, 2 * feat_dim)

    def forward(self, x, c):
        gb = self.to_gb(c)  # [B, 2*F]
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        beta  = beta.view(x.size(0), x.size(1), 1, 1)
        return x * (1.0 + gamma) + beta



# ----------------------------
# Discriminator (projection)
# ----------------------------

class DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect"))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="reflect"))
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1)) if in_ch != out_ch else None

    def forward(self, x):
        h = F.silu(self.conv1(x))
        h = F.silu(self.conv2(h))
        h = F.avg_pool2d(h, 2)
        x_skip = F.avg_pool2d(x, 2)
        if self.skip is not None:
            x_skip = self.skip(x_skip)
        return h + x_skip


class PairDiscriminator(nn.Module):
    def __init__(self, ch=64, x_ch=1, y_ch=2, im_size=img_size):
        super().__init__()
        in_ch = x_ch + y_ch +x_ch + y_ch

        layers = []
        layers.append(DBlock(in_ch, ch))       # 128->64
        layers.append(DBlock(ch, ch*2))        # 64->32
        layers.append(DBlock(ch*2, ch*4))      # 32->16
        layers.append(DBlock(ch*4, ch*8))      # 16->8
        self.blocks = nn.ModuleList(layers)

        self.conv_final = spectral_norm(nn.Conv2d(ch*8, ch*8, 3, padding=1))
        self.fc = spectral_norm(nn.Linear(ch*8, 1))

    def forward(self, x_pair):
        h = x_pair
        for blk in self.blocks:
            h = blk(h)
        h = F.silu(self.conv_final(h))
        feat = h.sum(dim=(2,3))
        out = self.fc(feat).squeeze(1)
        return out, feat


# ----------------------------
# Augmentations (turbulence-friendly)
# ----------------------------

def aug_turbulence(x):
    # x: [B,C,H,W]
    # periodic roll
    if torch.rand(()) < 0.9:
        B, C, H, W = x.shape
        dy = torch.randint(0, H, (1,), device=x.device).item()
        dx = torch.randint(0, W, (1,), device=x.device).item()
        x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    # flips
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(3,))
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(2,))
    # 90-degree rotations
    k = torch.randint(0, 4, (1,), device=x.device).item()
    if k:
        x = torch.rot90(x, k, dims=(2, 3))
    return x


# ----------------------------
# Losses
# ----------------------------

import torch
import torch.nn.functional as F

def grad_mag(x, eps=1e-6, periodic=True):
    """
    x: [B,1,H,W]
    returns g: [B,1,H,W] gradient magnitude
    """
    if periodic:
        dx = x - torch.roll(x, shifts=1, dims=-1)
        dy = x - torch.roll(x, shifts=1, dims=-2)
    else:
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]
        # pad back to H,W
        dx = F.pad(dx, (1,0,0,0), mode="replicate")
        dy = F.pad(dy, (0,0,1,0), mode="replicate")

    g = torch.sqrt(dx*dx + dy*dy + eps)
    return g

def soft_histogram(values, bin_centers, sigma, weights=None):
    """
    values: [B, N] (flattened samples)
    bin_centers: [K]
    sigma: float
    weights: [B, N] optional
    returns: pdf [B, K] normalized to sum=1
    """
    B, N = values.shape
    K = bin_centers.numel()

    v = values.unsqueeze(-1)                 # [B,N,1]
    c = bin_centers.view(1,1,K)              # [1,1,K]
    # Gaussian assignment
    a = torch.exp(-0.5 * ((v - c) / sigma)**2)  # [B,N,K]

    if weights is not None:
        a = a * weights.unsqueeze(-1)

    pdf = a.sum(dim=1)                       # [B,K]
    pdf = pdf / (pdf.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return pdf
def grad_pdf_loss(x_fake, x_real, *,
                  nbins=64,
                  qmin=0.001, qmax=0.999,
                  sigma_frac=0.5,
                  periodic=True,
                  detach_bins=True):
    """
    Returns: loss (scalar), pdf_fake [B,K], pdf_real [B,K], bin_centers [K]
    """
    # compute gradients in fp32 for stability
    gf = grad_mag(x_fake.float(), periodic=periodic)  # [B,1,H,W]
    gr = grad_mag(x_real.float(), periodic=periodic)

    vf = gf.flatten(1)  # [B, HW]
    vr = gr.flatten(1)

    # choose bins from real distribution (robust)
    # use global quantiles over batch to stabilize
    vr_all = vr.reshape(-1)
    lo = torch.quantile(vr_all, qmin)
    hi = torch.quantile(vr_all, qmax)

    # safety if lo==hi early
    hi = torch.maximum(hi, lo + 1e-6)

    bin_centers = torch.linspace(lo, hi, nbins, device=vr.device)

    # sigma as fraction of bin spacing
    delta = (hi - lo) / (nbins - 1)
    sigma = sigma_frac * delta

    if detach_bins:
        bin_centers = bin_centers.detach()

    pdf_f = soft_histogram(vf, bin_centers, sigma)
    pdf_r = soft_histogram(vr, bin_centers, sigma)

    # L1 between PDFs (stable). JS also works; L1 is fine to start.
    loss = (pdf_f - pdf_r).abs().mean()
    return loss, pdf_f, pdf_r, bin_centers


def d_hinge_loss(d_real, d_fake):
    return (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean())

def g_hinge_loss(d_fake):
    return (-d_fake.mean())

def r1_reg(d_out, x_real):
    # R1 gradient penalty on real images
    grad = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x_real,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return 0.5 * grad.pow(2).view(grad.size(0), -1).sum(dim=1).mean()


# ----------------------------
# Training step
# ----------------------------

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    # parameters
    for (name, p_ema) in ema_model.named_parameters():
        p = dict(model.named_parameters())[name]
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    # buffers (e.g., running stats if you later add norms)
    for (name, b_ema) in ema_model.named_buffers():
        b = dict(model.named_buffers())[name]
        b_ema.copy_(b)


def train(all_data, **kwargs):
    mean, std = estimate_log_stats(all_data['train'])
    ds_train = TurbulenceDataset(all_data['train'], augment=True, mean=mean, std=std)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)
    G, G_ema, D, mach_embed = train_gan(train_loader, **kwargs)
    return G, G_ema, D, mach_embed

def make_radial_bins_and_centers(H, W, device="cpu"):
    fy = torch.fft.fftfreq(H, d=1.0, device=device)    # cycles/pixel
    fx = torch.fft.rfftfreq(W, d=1.0, device=device)   # cycles/pixel (nonnegative)
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    kr = torch.sqrt(kx**2 + ky**2)                     # cycles/pixel, shape [H, W//2+1]

    kr_idx = kr * min(H, W)                            # "index units" ~1 per smallest mode
    bin_idx = torch.floor(kr_idx).long()
    nbins = int(bin_idx.max().item()) + 1

    # Bin centers in cycles/pixel: mean kr value per bin
    flat_kr = kr.reshape(-1)
    flat_idx = bin_idx.reshape(-1)

    counts = torch.bincount(flat_idx, minlength=nbins).clamp_min(1)
    sums   = torch.zeros(nbins, device=device, dtype=kr.dtype).scatter_add_(0, flat_idx, flat_kr)
    k_centers = sums / counts.to(kr.dtype)             # cycles/pixel

    return bin_idx, k_centers, nbins
def make_bispec_bin_pairs(nbins, kmin=1, kmax=None, max_pairs=64):
    """
    Return list of (b1,b2,b3) bin triplets satisfying b3 ~ b1+b2 in bin index space.
    For 32x32, nbins ~ 23. kmax default avoids the noisiest high-k bins.
    """
    if kmax is None:
        kmax = nbins - 2  # drop very highest bin

    pairs = []
    for b1 in range(kmin, kmax+1):
        for b2 in range(b1, kmax+1):
            b3 = b1 + b2
            if b3 <= kmax:
                pairs.append((b1,b2,b3))

    # subsample if too many
    if len(pairs) > max_pairs:
        step = max(1, len(pairs)//max_pairs)
        pairs = pairs[::step][:max_pairs]

    return torch.tensor(pairs, dtype=torch.long)  # [P,3]
def bispec_bicoherence(x, bin_idx, pairs, *, norm="ortho", eps=1e-8):
    """
    x: [B,1,H,W] or [B,H,W] real
    bin_idx: [H, W//2+1] long (rfft2 grid)
    pairs: [P,3] long (b1,b2,b3)

    returns: bicoh [B,P] in [0,1]-ish
    """
    if x.ndim == 4:
        x = x[:,0]
    B, H, W = x.shape

    # Remove mean so k=0 doesn't dominate
    x = x - x.mean(dim=(-2,-1), keepdim=True)

    X = torch.fft.rfft2(x, norm=norm)  # [B,H,W//2+1] complex
    # Flatten Fourier plane
    Xf = X.reshape(B, -1)              # [B,M]
    bflat = bin_idx.reshape(-1)        # [M]
    M = bflat.numel()

    # For each bin, collect indices of Fourier pixels in that bin
    # We'll build masks; for small 32x32 this is fine.
    nbins = int(bflat.max().item()) + 1
    masks = []
    for b in range(nbins):
        masks.append((bflat == b).nonzero(as_tuple=False).squeeze(1))  # [Mb]
    # Choose a representative set of k-vectors per bin by sampling a few modes
    # (stochastic estimate speeds up and regularizes)
    K = 32  # modes per bin; tune 16–64
    idx_per_bin = []
    rng = torch.rand  # uses global seed
    for b in range(nbins):
        inds = masks[b]
        if inds.numel() == 0:
            idx_per_bin.append(None)
            continue
        if inds.numel() <= K:
            idx_per_bin.append(inds)
        else:
            # random subset
            perm = torch.randperm(inds.numel(), device=inds.device)[:K]
            idx_per_bin.append(inds[perm])

    # Now estimate bicoherence for each (b1,b2,b3)
    out = []
    for (b1,b2,b3) in pairs.tolist():
        i1 = idx_per_bin[b1]
        i2 = idx_per_bin[b2]
        i3 = idx_per_bin[b3]
        if i1 is None or i2 is None or i3 is None:
            out.append(torch.zeros(B, device=X.device, dtype=torch.float32))
            continue

        # Sample matched counts
        n = min(i1.numel(), i2.numel(), i3.numel(), K)
        i1s = i1[:n]
        i2s = i2[:n]
        i3s = i3[:n]

        X1 = Xf[:, i1s]  # [B,n]
        X2 = Xf[:, i2s]
        X3 = Xf[:, i3s]

        # bispectrum estimate
        prod = X1 * X2 * torch.conj(X3)         # [B,n]
        Bhat = prod.mean(dim=1)                 # [B] complex

        # normalization (bicoherence-like)
        num = Bhat.abs()
        den = torch.sqrt(( (X1*X2).abs()**2 ).mean(dim=1) * (X3.abs()**2).mean(dim=1) + eps)
        out.append((num / den).float())         # [B]

    return torch.stack(out, dim=1)  # [B,P]
def bispectrum_loss(x_fake, x_real, bin_idx, pairs, eps=1e-8):
    # compute in fp32 for stability
    bf = bispec_bicoherence(x_fake.float(), bin_idx, pairs, eps=eps)
    br = bispec_bicoherence(x_real.float(), bin_idx, pairs, eps=eps)
    return (bf - br).abs().mean(), bf, br


def isotropic_power_spectrum(x, bin_idx, nbins, eps=1e-12):
    """
    x: [B, 1, H, W] (or [B,H,W]) real-valued
    bin_idx: [H, W//2+1] long
    returns: Pk [B, nbins]
    """
    if x.ndim == 4:
        x = x[:, 0]  # [B,H,W]
    B, H, W = x.shape

    # rfft2 -> [B,H,W//2+1]
    X = torch.fft.rfft2(x, norm="ortho")
    P = (X.real**2 + X.imag**2)  # power

    # Flatten frequency plane
    P = P.reshape(B, -1)                 # [B, HW']
    idx = bin_idx.reshape(-1)            # [HW']
    nb = nbins

    # Accumulate per-bin sums
    sums = torch.zeros(B, nb, device=x.device, dtype=P.dtype)
    sums.scatter_add_(1, idx.unsqueeze(0).expand(B, -1), P)

    # Count per bin
    counts = torch.bincount(idx, minlength=nb).to(x.device).to(P.dtype).clamp_min(1.0)  # [nb]
    Pk = sums / counts.unsqueeze(0)  # [B, nb]
    return Pk
def power_spectrum_loss(x_fake, x_real, bin_idx, nbins, log_eps=1e-12, remove_dc=True):
    """
    Compare isotropic power spectra in log-space.
    """
    # Optionally remove DC (mean) so spectrum focuses on fluctuations
    if remove_dc:
        x_fake = x_fake - x_fake.mean(dim=(-2, -1), keepdim=True)
        x_real = x_real - x_real.mean(dim=(-2, -1), keepdim=True)

    Pk_f = isotropic_power_spectrum(x_fake, bin_idx, nbins)  # [B,nb]
    Pk_r = isotropic_power_spectrum(x_real, bin_idx, nbins)  # [B,nb]

    #logPf = torch.log(Pk_f + log_eps)
    #logPr = torch.log(Pk_r + log_eps)
    logPf = Pk_f 
    logPr = Pk_r 

    # ignore k=0 bin if remove_dc (it should be tiny/noisy)
    if remove_dc and nbins > 1:
        logPf = logPf[:, 1:]
        logPr = logPr[:, 1:]

    return (logPf - logPr).abs().mean(), logPf, logPr

def fft2_phase_loss(x_fake, x_real, *, norm="ortho", remove_dc=True, eps=1e-8):
    """
    Phase-only FFT loss: compares F/|F| to keep only phase.
    """
    assert x_fake.shape == x_real.shape
    xf, xr = x_fake, x_real

    if remove_dc:
        xf = xf - xf.mean(dim=(-2, -1), keepdim=True)
        xr = xr - xr.mean(dim=(-2, -1), keepdim=True)

    Ff = torch.fft.fft2(xf, norm=norm)
    Fr = torch.fft.fft2(xr, norm=norm)

    Ff_u = Ff / (Ff.abs() + eps)
    Fr_u = Fr / (Fr.abs() + eps)

    diff = Ff_u - Fr_u
    return (diff.real**2 + diff.imag**2).mean().sqrt()

def laplacian(x):
    # x: [B,1,H,W]
    k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    return F.conv2d(x, k, padding=1)

def channelwise_mean(loss_fn, y_fake, y_real, **kwargs):
    # y_* : [B,2,H,W]
    losses = []
    for c in range(y_fake.size(1)):
        lf = loss_fn(y_fake[:,c:c+1], y_real[:,c:c+1], **kwargs)
        # loss_fn returns (loss, ...) in your helpers
        losses.append(lf[0] if isinstance(lf, (tuple,list)) else lf)
    return torch.stack(losses).mean()

def laplacian_multi(x):
    """
    x: [B,C,H,W]
    returns: [B,C,H,W]
    """
    k = torch.tensor([[0,1,0],
                      [1,-4,1],
                      [0,1,0]], device=x.device, dtype=x.dtype)
    k = k.view(1,1,3,3)                      # [1,1,3,3]
    k = k.expand(x.size(1), 1, 3, 3)         # [C,1,3,3]
    return F.conv2d(x, k, padding=1, groups=x.size(1))


def train_gan(
    loader,
    device="cuda",
    im_size=img_size,
    lr=lr,
    ch=64,
    x_ch=1,
    y_ch=2,
    r1_gamma=0.1,
    r1_every=32,
    ema_decay=0.999,
    steps=steps,
):
    G = main_net(in_channels=x_ch, out_channels=y_ch, base_channels=64).to(device)
    D = PairDiscriminator(ch=ch, x_ch=x_ch, y_ch=y_ch, im_size=im_size).to(device)

    #G_ema = UNetGenerator(in_ch=x_ch, out_ch=y_ch, base=64).to(device)
    G_ema = main_net(in_channels=x_ch, out_channels=y_ch, base_channels=64).to(device)
    G_ema.load_state_dict(G.state_dict())
    G_ema.eval()

    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.99))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.99))

    use_amp = (device == "cuda")
    scalerG = GradScaler("cuda", enabled=use_amp)
    scalerD = GradScaler("cuda", enabled=use_amp)

    data_iter = iter(loader)
    BIN_IDX, k_centers, NBINS = make_radial_bins_and_centers(im_size, im_size, device=device)
    PAIRS = make_bispec_bin_pairs(NBINS, kmin=1, kmax=min(NBINS-2, 10), max_pairs=64).to(device)

    import tqdm
    for step in tqdm.tqdm(range(1, steps+1)):
        try:
            x_in, y_real = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x_in, y_real = next(data_iter)

        x_in   = x_in.to(device, non_blocking=True).float()     # [B,1,H,W]
        y_real = y_real.to(device, non_blocking=True).float()   # [B,2,H,W]

        # optional paired augmentation
        if True:
            # roll/flip/rot consistent across x and y
            cat = torch.cat([x_in, y_real], dim=1)
            cat = aug_turbulence(cat)
            x_in, y_real = cat[:, :x_ch], cat[:, x_ch:]

        # ---------------- D step ----------------
        optD.zero_grad(set_to_none=True)

        if step % r1_every == 0:
            y_real.requires_grad_(True)

        with autocast(device, enabled=use_amp):
            y_fake = G(x_in).detach()
            pair_real = torch.cat([x_in, y_real, laplacian_multi(x_in), laplacian_multi(y_real)], dim=1)
            pair_fake = torch.cat([x_in, y_fake, laplacian_multi(x_in), laplacian_multi(y_fake)], dim=1)

            d_real, _ = D(pair_real)
            d_fake, _ = D(pair_fake)

            #d_real, _ = D(torch.cat([x_in, y_real], dim=1))
            #d_fake, _ = D(torch.cat([x_in, y_fake], dim=1))

            lossD = d_hinge_loss(d_real, d_fake)

            if (step % r1_every) == 0:
                # R1 on the *pair* or just on y_real; simplest: y_real
                r1 = r1_reg(d_real, y_real)
                lossD = lossD + r1_gamma * r1

        scalerD.scale(lossD).backward()
        scalerD.step(optD)
        scalerD.update()

        if step % r1_every == 0:
            y_real.requires_grad_(False)

        # ---------------- G step ----------------
        optG.zero_grad(set_to_none=True)

        with autocast(device, enabled=use_amp):
            y_fake = G(x_in)
            pair_fake = torch.cat([x_in, y_fake, laplacian_multi(x_in), laplacian_multi(y_fake)], dim=1)
            d_fake, _ = D(pair_fake)
            #d_fake, _ = D(torch.cat([x_in, y_fake], dim=1))

            adv = g_hinge_loss(d_fake)
            l1  = (y_fake - y_real).abs().mean()

            # your spectral-ish losses, averaged over channels
            ps   = channelwise_mean(lambda a,b,**kw: power_spectrum_loss(a.float(), b.float(), BIN_IDX, NBINS, remove_dc=True),
                                    y_fake, y_real)
            gpdf = channelwise_mean(lambda a,b,**kw: grad_pdf_loss(a, b, nbins=64),
                                    y_fake, y_real)
            ph   = channelwise_mean(lambda a,b,**kw: (fft2_phase_loss(a.float(), b.float(), eps=1e-4),),
                                    y_fake, y_real)
            bi   = channelwise_mean(lambda a,b,**kw: bispectrum_loss(a, b, BIN_IDX, PAIRS),
                                    y_fake, y_real)

            # weights: you’ll retune these
            lossG = adv + 0.1*l1 + 0.5*ps + 0.8*bi + 0.8*gpdf + 0.1*ph

        scalerG.scale(lossG).backward()
        scalerG.step(optG)
        scalerG.update()

        update_ema(G_ema, G, decay=ema_decay)

        if step % 100 == 0:
            print(f"step {step:7d}  lossD {lossD.item():.4f}  lossG {lossG.item():.4f}")
        if step % 20 == 0:
            fig,axes=plt.subplots(3,4,figsize=(12,8))
            ax0,ax1,ax2=axes
            def f(arr):
                return arr.cpu().detach().numpy()
            plot_three(f(x_in[0][0]),  f(x_in[0][0]),title='T', axs=ax0, fig=fig)
            plot_three(f(y_real[0][0]),f(y_fake[0][0]),title='E', axs=ax1, fig=fig)
            plot_three(f(y_real[0][1]),f(y_fake[0][1]),title='B', axs=ax2, fig=fig)
            fig.tight_layout()
            fig.savefig('%s/plots/p79g_%04d.png'%(os.environ['HOME'],step))



    return G, G_ema, D

import matplotlib as mpl
from scipy.stats import pearsonr
import dtools_global.vis.pcolormesh_helper as pch

def plot_three(Etarget,Eguess,fig=None,axs=None, title='', floating=True):
    if hasattr(Eguess, 'cpu'):
        Eguess=Eguess.cpu()
    if hasattr(Etarget,'cpu'):
        Etarget=Etarget.cpu()
    Emin = min([Etarget.min(), Eguess.min()])
    Emax = max([Etarget.max(), Eguess.max()])
    if not floating:
        #Enorm = mpl.colors.Normalize(vmin=Emin,vmax=Emax)
        Enorm = mpl.colors.SymLogNorm(1.0,vmin=Emin,vmax=Emax)
    else:
        Enorm = None
    ppp=axs[0].imshow(Etarget,norm=Enorm)
    fig.colorbar(ppp,ax=axs[0])
    axs[0].set(title='%s actual'%title, xlabel='x [pixel]', ylabel = 'y [pixel]')
    L2 = -1#F.mse_loss(Etarget,Eguess)
    ppp=axs[1].imshow(Eguess,norm=Enorm)
    fig.colorbar(ppp,ax=axs[1])
    axs[1].set(title='%s predict %0.2e'%(title,L2), xlabel='x [pixel]', ylabel = 'y [pixel]')
    er = pearsonr( Eguess.flatten(), Etarget.flatten())[0]
    fig.colorbar(ppp,ax=axs[2])
    axs[2].set(title='pearson %0.4f'%er)
    E1 = Etarget.flatten()
    E2 = Eguess.flatten()
    pch.simple_phase(E1,E2,ax=axs[2], colorbar=False)
    axs[2].plot( [Emin,Emax],[Emin,Emax],c='k')
    axs[2].set(xlabel='Actual pixel value',ylabel='Predicted')
    import dtools_global.math.power_spectrum as ps
    if len(axs)>=4:
        #power_guess = torch_power.powerspectrum(Eguess)
        #power_target = torch_power.powerspectrum(Etarget)
        power_guess = ps.powerspectrum(Eguess)
        power_target = ps.powerspectrum(Etarget)
        axs[3].plot( power_guess.kcen, power_guess.avgpower, c='r')
        axs[3].plot( power_target.kcen, power_target.avgpower, c='k')
        axs[3].set(xscale='log',yscale='log', title='Power spectrum %s'%title, xlabel='k', ylabel='power')
    if len(axs)==5:
        cross=ps.cross_spectrum(Eguess.detach().numpy(), Etarget.detach().numpy())
        axs[4].plot(power_target.kcen, power_target.avgpower, c='k')
        axs[4].plot(cross.kcen, cross.avgpower, c='r')
        axs[4].set(xscale='log',yscale='log', title='Cross spectrum %s'%title, xlabel='k', ylabel='power')

