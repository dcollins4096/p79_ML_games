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

idd = 1
what = "First stab at a Generative Adversarial Network"

dirpath = "/home/dcollins/repos/p79_ML_games/p79d_kyes/datasets/"
#dirpath = "./"

fname_train = "p79d_subsets_S128_N1_xyz_suite7vs_first.h5"
fname_valid = "p79d_subsets_S128_N1_xyz_suite7vs_second.h5"

ntrain = 14000
nvalid = 100
ntest = 5000
downsample = None
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Hyperparameters
epochs = 500
lr = 1e-4
batch_size = 64
lr_schedule = [1000]
weight_decay = 0.01

# Hybrid architecture parameters
img_size = 128
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


class TurbulenceDataset(Dataset):
    def __init__(self, all_data, quan, augment=False, mean=0.0, std=1.0):
        self.quan = quan
        self.all_data = all_data
        self.augment = augment
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self):
        return self.all_data.shape[0]   # or len(self.all_data)

    def __getitem__(self, idx):
        data = self.all_data[idx][0:1].float()  # [1,H,W]

        # image transform FIRST (before aug or after; both can work)
        data = torch.log1p(torch.clamp_min(data, 0.0))
        data = (data - self.mean) / (self.std + 1e-6)
        data = data.clamp(-5, 5)  # helps early stability

        # augments
        if self.augment:
            H, W = data[0].shape
            dy = torch.randint(0, H, (1,)).item()
            dx = torch.randint(0, W, (1,)).item()
            data = torch.roll(data, shifts=(dy, dx), dims=(-2, -1))
            if torch.rand(1) > 0.5: data = torch.flip(data, dims=[-1])
            if torch.rand(1) > 0.5: data = torch.flip(data, dims=[-2])

        ms = float(self.quan['Ms_act'][idx])
        ms = math.log(ms)  # ok, but…
        return data, torch.tensor(ms, dtype=torch.float32)


# ----------------------------
# Utilities: conditioning
# ----------------------------

class MachEmbed(nn.Module):
    """Embed a scalar Mach number into a conditioning vector."""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

    def forward(self, mach):  # mach: [B]
        return self.net(mach.view(-1, 1))


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
# Generator
# ----------------------------

class GBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film1 = FiLM(cond_dim, in_ch)
        self.film2 = FiLM(cond_dim, out_ch)
        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, c):
        h = self.film1(x, c)
        h = F.silu(h)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.conv1(h)

        h = self.film2(h, c)
        h = F.silu(h)
        h = self.conv2(h)

        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if self.skip is not None:
            x_up = self.skip(x_up)
        return h + x_up


class Generator(nn.Module):
    def __init__(self, z_dim=128, cond_dim=128, ch=64, out_ch=1, im_size=128):
        super().__init__()
        assert im_size in [64, 128, 256], "Adjust blocks if using other sizes."
        self.z_dim = z_dim
        self.im_size = im_size

        self.fc = nn.Linear(z_dim, 4 * 4 * (ch * 8))
        self.blocks = nn.ModuleList()

        # 4->8->16->32->64->128 (and optionally 256)
        self.blocks.append(GBlock(ch*8, ch*8, cond_dim))  # 4->8
        self.blocks.append(GBlock(ch*8, ch*4, cond_dim))  # 8->16
        self.blocks.append(GBlock(ch*4, ch*2, cond_dim))  # 16->32
        self.blocks.append(GBlock(ch*2, ch*1, cond_dim))  # 32->64
        if im_size >= 128:
            self.blocks.append(GBlock(ch*1, ch//2, cond_dim))  # 64->128
            final_ch = ch//2
        else:
            final_ch = ch*1
        if im_size >= 256:
            self.blocks.append(GBlock(final_ch, final_ch//2, cond_dim))  # 128->256
            final_ch = final_ch//2

        self.to_img = nn.Conv2d(final_ch, out_ch, 3, padding=1)

    def forward(self, z, c):
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        for blk in self.blocks:
            h = blk(h, c)
        h = F.silu(h)
        x = self.to_img(h)
        return x  # keep as real-valued (no tanh) since we standardize data


# ----------------------------
# Discriminator (projection)
# ----------------------------

class DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1)) if in_ch != out_ch else None

    def forward(self, x):
        h = F.silu(self.conv1(x))
        h = F.silu(self.conv2(h))
        h = F.avg_pool2d(h, 2)
        x_skip = F.avg_pool2d(x, 2)
        if self.skip is not None:
            x_skip = self.skip(x_skip)
        return h + x_skip


class Discriminator(nn.Module):
    def __init__(self, cond_dim=128, ch=64, in_ch=1, im_size=128):
        super().__init__()
        assert im_size in [64, 128, 256], "Adjust blocks if using other sizes."
        self.im_size = im_size

        layers = []
        # in -> ch -> 2ch -> 4ch -> 8ch ...
        layers.append(DBlock(in_ch, ch))       # 128->64
        layers.append(DBlock(ch, ch*2))        # 64->32
        layers.append(DBlock(ch*2, ch*4))      # 32->16
        layers.append(DBlock(ch*4, ch*8))      # 16->8
        if im_size >= 256:
            layers.append(DBlock(ch*8, ch*8)) # 256->128 extra stage
        self.blocks = nn.ModuleList(layers)

        self.conv_final = spectral_norm(nn.Conv2d(ch*8, ch*8, 3, padding=1))
        self.fc = spectral_norm(nn.Linear(ch*8, 1))

        # Projection: <phi(x), c_embed>
        self.proj = spectral_norm(nn.Linear(cond_dim, ch*8))

    def forward(self, x, c):
        h = x
        for blk in self.blocks:
            h = blk(h)
        h = F.silu(self.conv_final(h))
        # global sum pool
        feat = h.sum(dim=(2, 3))  # [B, F]
        out = self.fc(feat).squeeze(1)  # [B]

        proj = (self.proj(c) * feat).sum(dim=1)  # [B]
        return out + proj, feat


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
    ds_train = TurbulenceDataset(all_data['train'], all_data['quantities']['train'], augment=True, mean=mean, std=std)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    G, G_ema, D, mach_embed = train_gan(train_loader, **kwargs)
    return G, G_ema, D, mach_embed


def train_gan(
    loader,  # yields (x, mach)
    device="cuda",
    im_size=128,
    z_dim=128,
    cond_dim=128,
    lr=lr,
    ch=64,
    out_ch=1,
    r1_gamma=0.1,
    r1_every=32,
    ema_decay=0.999,
    steps=100000
):
    mach_embed = MachEmbed(embed_dim=cond_dim).to(device)
    G = Generator(z_dim=z_dim, cond_dim=cond_dim, ch=ch, out_ch=out_ch, im_size=im_size).to(device)
    D = Discriminator(cond_dim=cond_dim, ch=ch, in_ch=out_ch, im_size=im_size).to(device)

    G_ema = Generator(z_dim=z_dim, cond_dim=cond_dim, ch=ch, out_ch=out_ch, im_size=im_size).to(device)
    G_ema.load_state_dict(G.state_dict())
    G_ema.eval()

    optG = torch.optim.Adam(list(G.parameters()) + list(mach_embed.parameters()), lr=lr, betas=(0.0, 0.99))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.99))

    use_amp = (device == "cuda")
    scalerG = GradScaler(device, enabled=use_amp)
    scalerD = GradScaler(device, enabled=use_amp)

    data_iter = iter(loader)

    import tqdm
    steps = list(range(1, steps + 1))
    for step in tqdm.tqdm(steps):
        try:
            x_real, mach = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x_real, mach = next(data_iter)

        x_real = x_real.to(device, non_blocking=True).float()
        mach = mach.to(device, non_blocking=True).float()

        # --- Discriminator ---
        optD.zero_grad(set_to_none=True)
        x_real = aug_turbulence(x_real)

        x_real.requires_grad_(True)

        with autocast(device, enabled=use_amp):
            c = mach_embed(mach)

            z = torch.randn(x_real.size(0), z_dim, device=device)
            x_fake = G(z, c).detach()
            #x_fake = aug_turbulence(x_fake)

            d_real, _ = D(x_real, c)
            d_fake, _ = D(x_fake, c)

            lossD = d_hinge_loss(d_real, d_fake)

            # R1 regularization (every r1_every steps)
            if (step % r1_every) == 0:
                r1 = r1_reg(d_real, x_real)
                lossD = lossD + r1_gamma * r1

        scalerD.scale(lossD).backward()
        scalerD.step(optD)
        scalerD.update()

        x_real.requires_grad_(False)

        # --- Generator ---
        optG.zero_grad(set_to_none=True)
        with autocast(device, enabled=use_amp):
            c = mach_embed(mach)  # same batch conditioning
            z = torch.randn(x_real.size(0), z_dim, device=device)
            x_fake = G(z, c)
            #x_fake = aug_turbulence(x_fake)

            d_fake, _ = D(x_fake, c)
            lossG = g_hinge_loss(d_fake)

        scalerG.scale(lossG).backward()
        scalerG.step(optG)
        scalerG.update()

        update_ema(G_ema, G, decay=ema_decay)

        if step % 100 == 0:
            print(f"step {step:7d}  lossD {lossD.item():.4f}  lossG {lossG.item():.4f}")
        if step % 20 == 0:
            import matplotlib.pyplot as plt
            xr = x_real[0,0].detach().float().cpu().numpy()
            xf = x_fake[0,0].detach().float().cpu().numpy()
            vmin = min(xr.min(), xf.min())
            vmax = max(xr.max(), xf.max())

            plt.figure(figsize=(8,4))
            plt.subplot(2,2,1); plt.imshow(xr, origin="lower", vmin=vmin, vmax=vmax); plt.title("REAL"); plt.colorbar()
            plt.subplot(2,2,2); plt.imshow(xf, origin="lower", vmin=vmin, vmax=vmax); plt.title("FAKE"); plt.colorbar()
            a = sorted(xr.flatten())
            plt.subplot(2,2,3); plt.plot( a, np.arange(len(a))/len(a))
            a = sorted(xf.flatten())
            plt.subplot(2,2,4); plt.plot( a, np.arange(len(a))/len(a))
            plt.tight_layout()
            oot=f"%s/plots/debug_real_fake_{step:04}.png"%os.environ['HOME']
            plt.savefig(oot, dpi=150)
            plt.close()

    return G, G_ema, D, mach_embed

