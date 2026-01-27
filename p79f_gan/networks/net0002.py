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
import h5py

idd = 2
what = "GAN with lines"

#dirpath = "/home/dcollins/repos/p79_ML_games/p79d_kyes/datasets/"
#dirpath = "./"

fname = "datasets/random_lines_128_fixed.h5"
fname = "datasets/random_lines_128_12zones.h5"
fname = "datasets/random_lines_128_32zones.h5"
#fname = "datasets/random_lines_128_64zones.h5"

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
img_size = 32
cnn_output_size = 64  # After CNN pooling: 128 → 64
patch_size = 8  # Smaller patches on reduced resolution
embed_dim = 384
depth = 6
num_heads = 6
mlp_ratio = 4.0
dropout = 0.1


def load_data():
    fptr = h5py.File(fname,'r')
    all_data={}
    all_data['images'] =torch.tensor(fptr['images'][()])
    all_data['x0'] =    torch.tensor(fptr['x0'][()])
    all_data['y0'] =    torch.tensor(fptr['y0'][()])
    all_data['theta'] = torch.tensor(fptr['theta'][()])
    all_data['width'] = torch.tensor(fptr['width'][()])


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


import h5py
import torch
from torch.utils.data import Dataset

class LineDataset(Dataset):
    def __init__(self, all_data, img_size=img_size, to_minus1_1=True):
        self.all_data = all_data
        self.img_size = img_size
        self.to_minus1_1 = to_minus1_1

    def __len__(self):
        return self.all_data['images'].shape[0]

    def __getitem__(self, idx):
        x0    = self.all_data['x0'][idx].float()
        y0    = self.all_data['y0'][idx].float()
        theta = self.all_data['theta'][idx].float()
        width = self.all_data['width'][idx].float()

        # image: [128,128] -> [1,128,128]
        img = self.all_data['images'][idx].float()
        if img.ndim == 2:
            img = img.unsqueeze(0)

        # map {0,1} -> {-1,+1} (recommended for GAN with tanh)
        if self.to_minus1_1:
            img = img * 2.0 - 1.0

        # normalize params to roughly [-1,1]
        # x0,y0 are in pixel coords [0,127]
        x0n = (x0 / (self.img_size - 1.0)) * 2.0 - 1.0
        y0n = (y0 / (self.img_size - 1.0)) * 2.0 - 1.0

        # width: pick a scale based on your generator range; if widths are ~[1,6], this is fine
        # map to [-1,1] assuming width_range=(1,6); adjust if different
        wmin, wmax = 1.0, 6.0
        wn = (width - wmin) / (wmax - wmin) * 2.0 - 1.0

        # theta -> (cos,sin)
        cth = torch.cos(theta)
        sth = torch.sin(theta)

        cond = torch.stack([x0n, y0n, cth, sth, wn], dim=0)  # [5]
        return cond, img


# ----------------------------
# Utilities: conditioning
# ----------------------------

class ParamEmbed(nn.Module):
    """Embed conditioning vector into cond_dim."""
    def __init__(self, in_dim=5, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

    def forward(self, cond):  # cond: [B,5]
        return self.net(cond)


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
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect")
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="reflect")
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
        assert im_size in [32, 64, 128, 256], "im_size must be one of {32,64,128,256}"
        self.z_dim = z_dim
        self.im_size = im_size

        # start from 4x4
        self.fc = nn.Linear(z_dim, 4 * 4 * (ch * 8))

        blocks = []
        # Always: 4->8 and 8->16
        blocks.append(GBlock(ch*8, ch*8, cond_dim))  # 4->8
        blocks.append(GBlock(ch*8, ch*4, cond_dim))  # 8->16

        if im_size >= 32:
            blocks.append(GBlock(ch*4, ch*2, cond_dim))  # 16->32
            final_ch = ch*2
        if im_size >= 64:
            blocks.append(GBlock(final_ch, ch*1, cond_dim))  # 32->64
            final_ch = ch*1
        if im_size >= 128:
            blocks.append(GBlock(final_ch, ch//2, cond_dim))  # 64->128
            final_ch = ch//2
        if im_size >= 256:
            blocks.append(GBlock(final_ch, max(final_ch//2, 8), cond_dim))  # 128->256
            final_ch = max(final_ch//2, 8)

        self.blocks = nn.ModuleList(blocks)

        # Use reflect padding to avoid border cheats
        self.to_img = nn.Conv2d(final_ch, out_ch, 3, padding=1, padding_mode="reflect")

    def forward(self, z, c):
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        for blk in self.blocks:
            h = blk(h, c)
        h = F.silu(h)
        x = self.to_img(h)
        return torch.tanh(x)  # if your real images are scaled to [-1,1]

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


class Discriminator(nn.Module):
    def __init__(self, cond_dim=128, ch=64, in_ch=1, im_size=img_size):
        super().__init__()
        assert im_size in [32,64, 128, 256], "Adjust blocks if using other sizes."
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
    return x
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
    ds_train = LineDataset(all_data)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)
    G, G_ema, D, cond_embed = train_gan(train_loader, **kwargs)
    return G, G_ema, D, cond_embed


def train_gan(
    loader,  # yields (x, mach)
    device="cuda",
    im_size=img_size,
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
    cond_embed = ParamEmbed(in_dim=5, embed_dim=cond_dim).to(device)
    G = Generator(z_dim=z_dim, cond_dim=cond_dim, ch=ch, out_ch=out_ch, im_size=im_size).to(device)
    D = Discriminator(cond_dim=cond_dim, ch=ch, in_ch=out_ch, im_size=im_size).to(device)

    G_ema = Generator(z_dim=z_dim, cond_dim=cond_dim, ch=ch, out_ch=out_ch, im_size=im_size).to(device)
    G_ema.load_state_dict(G.state_dict())
    G_ema.eval()

    optG = torch.optim.Adam(list(G.parameters()) + list(cond_embed.parameters()), lr=lr, betas=(0.0, 0.99))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.99))

    use_amp = (device == "cuda")
    scalerG = GradScaler("cuda", enabled=use_amp)
    scalerD = GradScaler("cuda", enabled=use_amp)

    data_iter = iter(loader)

    import tqdm
    steps = list(range(1, steps + 1))
    for step in tqdm.tqdm(steps):
        try:
            cond, x_real = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            cond, x_real = next(data_iter)

        x_real = x_real.to(device, non_blocking=True).float()
        cond = cond.to(device, non_blocking=True).float()

        # --- Discriminator ---
        optD.zero_grad(set_to_none=True)
        #x_real = aug_turbulence(x_real)

        if step%r1_every == 0:
            x_real.requires_grad_(True)

        with autocast(device, enabled=use_amp):
            c = cond_embed(cond)

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

        if step%r1_every == 0:
            x_real.requires_grad_(False)

        # --- Generator ---
        optG.zero_grad(set_to_none=True)
        with autocast(device, enabled=use_amp):
            c = cond_embed(cond)  # same batch conditioning
            z = torch.randn(x_real.size(0), z_dim, device=device)
            x_fake = G(z, c)
            #x_fake = aug_turbulence(x_fake)

            d_fake, _ = D(x_fake, c)
            l1 = (x_fake - x_real).abs().mean()
            lossG = g_hinge_loss(d_fake) + 10.0*l1

        scalerG.scale(lossG).backward()
        scalerG.step(optG)
        scalerG.update()

        update_ema(G_ema, G, decay=ema_decay)

        if step % 100 == 0:
            print(f"step {step:7d}  lossD {lossD.item():.4f}  lossG {lossG.item():.4f}")
        if step % 2 == 0:
            import matplotlib.pyplot as plt
            xr = x_real[0,0].detach().float().cpu().numpy()
            xf = x_fake[0,0].detach().float().cpu().numpy()
            vmin = min(xr.min(), xf.min())
            vmax = max(xr.max(), xf.max())

            plt.figure(figsize=(8,4))
            plt.subplot(2,2,1); plt.imshow(xr, origin="lower"); plt.title("REAL"); plt.colorbar()
            plt.subplot(2,2,2); plt.imshow(xf, origin="lower"); plt.title("FAKE"); plt.colorbar()
            a = sorted(xr.flatten())
            plt.subplot(2,2,3); plt.plot( a, np.arange(len(a))/len(a))
            a = sorted(xf.flatten())
            plt.subplot(2,2,4); plt.plot( a, np.arange(len(a))/len(a))
            plt.tight_layout()
            oot=f"%s/plots/debug_real_fake_{step:04}.png"%os.environ['HOME']
            plt.savefig(oot, dpi=150)
            plt.close()
            def param_norm(m):
                s = 0.0
                for p in m.parameters():
                    s += p.detach().float().pow(2).sum().item()
                return s**0.5

            def grad_norm(m):
                s = 0.0
                for p in m.parameters():
                    if p.grad is None: 
                        continue
                    g = p.grad.detach().float()
                    s += g.pow(2).sum().item()
                return s**0.5

            if step % 20 == 0:
                print("||G||:", param_norm(G), "||∇G||:", grad_norm(G))
                print("||D||:", param_norm(D), "||∇D||:", grad_norm(D))
            if step % 20 == 0:
                print("D(real) mean:", d_real.detach().mean().item(),
                      "D(fake) mean:", d_fake.detach().mean().item())
                print("hinge active real:", (1.0 - d_real.detach() > 0).float().mean().item(),
                      "fake:", (1.0 + d_fake.detach() > 0).float().mean().item())

    return G, G_ema, D, cond_embed

