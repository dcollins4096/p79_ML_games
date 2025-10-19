import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time
import datetime
import healpy as hp
import random
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import pdb
import loader
from scipy.ndimage import gaussian_filter
import torch_power


idd = 201
what = "Vision Transformer"

fname_train = "p79d_subsets_S256_N5_xyz_down_12823456_first.h5"
fname_valid = "p79d_subsets_S256_N5_xyz_down_12823456_second.h5"
#ntrain = 2000
#ntrain = 1000 #ntrain = 600
#ntrain = 20
ntrain = 3000
#nvalid=3
#ntrain = 10
nvalid=30
downsample = 64
#device = device or ("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
#epochs  = 20
epochs = 200
lr = 1e-3
#lr = 1e-4
batch_size=64
lr_schedule=[100]
weight_decay = 1e-3
fc_bottleneck=True
def load_data():

    print('read the data')
    train= loader.loader(fname_train,ntrain=ntrain, nvalid=nvalid)
    valid= loader.loader(fname_valid,ntrain=1, nvalid=nvalid)
    all_data={'train':train['train'],'valid':valid['valid'], 'test':valid['test'], 'quantities':{}}
    all_data['quantities']['train']=train['quantities']['train']
    all_data['quantities']['valid']=valid['quantities']['valid']
    all_data['quantities']['test']=valid['quantities']['test']
    print('done')
    return all_data

def thisnet():

    model = main_net()

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model

def train(model,all_data):
    trainer(model,all_data,epochs=epochs,lr=lr,batch_size=batch_size, weight_decay=weight_decay, lr_schedule=lr_schedule)

import torch
import torch.nn.functional as F

def downsample_avg(x, M):
    if x.ndim == 2:   # [N, N]
        x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, N, N]
        out = F.adaptive_avg_pool2d(x, (M, M))
        return out.squeeze(0).squeeze(0) # -> [M, M]
    elif x.ndim == 4: # [B, C, N, N]
        return F.adaptive_avg_pool2d(x, (M, M))
    else:
        raise ValueError("Input must be [N, N] or [B, C, N, N]")


# ---------------------------
# Dataset with input normalization
# ---------------------------
import torchvision.transforms.functional as TF
import random
class SphericalDataset(Dataset):
    def __init__(self, all_data, rotation_prob = 0.0):
        self.rotation_prob = rotation_prob
        if downsample:
            self.all_data=downsample_avg(all_data,downsample)
        else:
            self.all_data=all_data
    def __len__(self):
        return self.all_data.size(0)

    def __getitem__(self, idx):
        #return self.data[idx], self.targets[idx]
        theset = self.all_data[idx]
        if random.uniform(0,1) < self.rotation_prob:
            angle = random.uniform(-90,90)
            theset = TF.rotate(theset,angle)
        return theset[0], theset

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed=8675309):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Train / Eval
# ---------------------------
def trainer(
    model,
    all_data,
    epochs=200,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    warmup_frac=0.05,
    lr_schedule=[900],
    plot_path=None
):
    set_seed()

    ds_train = SphericalDataset(all_data['train'], rotation_prob=model.rotation_prob)
    ds_val   = SphericalDataset(all_data['valid'], rotation_prob=model.rotation_prob)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(ds_val,   batch_size=max(64, batch_size), shuffle=False, drop_last=False)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    print("Total Steps", total_steps)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_schedule, #[100,300,600],  # change after N and N+M steps
        gamma=0.1             # multiply by gamma each time
    )

    best_val = float("inf")
    best_state = None
    patience = 25
    bad_epochs = 0

    train_curve, val_curve = [], []
    t0 = time.time()
    verbose=False
    save_err_Cross = -1
    if model.err_Cross > 0:
        save_err_Cross = model.err_Cross
        model.err_Cross = torch.tensor(0.0)


    for epoch in range(1, epochs+1):
        if epoch > 50 and save_err_Cross>0:
            model.err_Cross = save_err_Cross
        model.train()
        if verbose:
            print("Epoch %d"%epoch)
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            if verbose:
                print("  model")
            if 1:
                preds = model(xb)
                if verbose:
                    print("  crit")

                #loss  = model.criterion(preds, yb[:,0:1,:,:])
                loss  = model.criterion(preds, yb)

            if verbose:
                print("  scale backward")
            loss.backward()

            if verbose:
                print("  steps")
            optimizer.step()

            running += loss.item() * xb.size(0)

        scheduler.step()
        train_loss = running / len(ds_train)
        train_curve.append(train_loss)

        # validate
        if verbose:
            print("  valid")
        model.eval()
        with torch.no_grad():
            vtotal = 0.0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                #vloss = model.criterion(preds, yb[:,0:1,:,:])
                vloss = model.criterion(preds, yb)
                vtotal += vloss.item() * xb.size(0)
            val_loss = vtotal / len(ds_val)
            val_curve.append(val_loss)
        #model.train_curve = torch.tensor(train_curve)
        #model.val_curve = torch.tensor(val_curve)
        model.train_curve[epoch-1] = train_loss
        model.val_curve[epoch-1] = val_loss

        # early stopping
        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        # progress line
        now = time.time()
        time_per_epoch = (now - t0) / epoch
        secs_left = time_per_epoch * (epochs - epoch)
        etad = datetime.datetime.fromtimestamp(now + secs_left)
        eta = etad.strftime("%H:%M:%S")
        nowdate = datetime.datetime.fromtimestamp(now)
        #lr = scheduler.get_last_lr()[0]
        lr = optimizer.param_groups[0]['lr']


        def format_time(seconds):
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        elps = format_time( now-t0)
        rem  = format_time(secs_left)

        #model.train_curve = torch.tensor(train_curve)
        #model.val_curve = torch.tensor(val_curve)

        print(f"[{epoch:3d}/{epochs}] net{idd:d}  train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {lr:.2e} | bad {bad_epochs:02d} | ETA {eta} | Remain {rem} | Sofar {elps}")
        if nowdate.day - etad.day != 0:
            print('tomorrow')

        if bad_epochs >= patience and False:
            print(f"Early stopping at epoch {epoch}. Best val {best_val:.4f}.")
            print('disabled')
            #break

    return model
    # restore best
    #if best_state is not None:
    #    model.load_state_dict(best_state)

    # quick plot (optional)

def plot_loss_curve(model):
    plt.clf()
    plt.plot(model.train_curve.cpu(), label="train")
    plt.plot(model.val_curve.cpu(),   label="val")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/plots/errtime_net%04d"%(os.environ['HOME'], model.idd))

def power_spectrum_delta(guess,target):
    T_guess = torch_power.powerspectrum(guess)
    T_target = torch_power.powerspectrum(target)
    output = torch.mean( torch.abs(torch.log(T_guess.avgpower/(T_target.avgpower+1e-8))))
    return output

def power_spectra_crit(guess,target):
    err_T = power_spectrum_delta(guess[:,0:1,:,:], target[:,0:1,:,:])
    err_E = power_spectrum_delta(guess[:,1:2,:,:], target[:,1:2,:,:])
    err_B = power_spectrum_delta(guess[:,2:3,:,:], target[:,2:3,:,:])
    return err_T+err_E+err_B

def cross_spectrum_delta(guess,target):
    T_guess = torch_power.crossspectrum(guess, target)
    T_target = torch_power.powerspectrum(target)
    num = torch.clamp(torch.abs(T_guess.avgpower), min=1e-12)
    den = torch.clamp(T_target.avgpower, min=1e-12)
    output = torch.mean(torch.abs(torch.log(num / den)))
    if (T_guess.avgpower < 0).any():
            negative = torch.mean(torch.abs(T_guess.avgpower[T_guess.avgpower < 0]))
    else:
            negative = 0.0
    return output+negative

def cross_spectrum_cosine(guess, target):
    G = torch.fft.fftn(guess, dim=(-2, -1))
    T = torch.fft.fftn(target, dim=(-2, -1))
    num = torch.sum(G * torch.conj(T)).real
    denom = torch.sqrt(torch.sum(torch.abs(G)**2) * torch.sum(torch.abs(T)**2))
    return 1 - num / (denom + 1e-12)

def cross_spectra_crit(guess,target):
    err_T = cross_spectrum_cosine(guess[:,0:1,:,:], target[:,0:1,:,:])
    err_E = cross_spectrum_cosine(guess[:,1:2,:,:], target[:,1:2,:,:])
    err_B = cross_spectrum_cosine(guess[:,2:3,:,:], target[:,2:3,:,:])
    return err_T+err_E+err_B

import bispectrum
def bispectrum_crit(guess,target):
    nsamples=100
    T_guess = bispectrum.compute_bispectrum_torch(guess[:,0:1,:,:]  ,nsamples=nsamples)[0]
    E_guess = bispectrum.compute_bispectrum_torch(guess[:,1:2,:,:]  ,nsamples=nsamples)[0]
    B_guess = bispectrum.compute_bispectrum_torch(guess[:,2:3,:,:]  ,nsamples=nsamples)[0]
    T_target = bispectrum.compute_bispectrum_torch(target[:,0:1,:,:],nsamples=nsamples)[0]
    E_target = bispectrum.compute_bispectrum_torch(target[:,1:2,:,:],nsamples=nsamples)[0]
    B_target = bispectrum.compute_bispectrum_torch(target[:,2:3,:,:],nsamples=nsamples)[0]
    dT = torch.mean(torch.abs(torch.log(torch.abs( T_guess / T_target))))
    dE = torch.mean(torch.abs(torch.log(torch.abs( E_guess / E_target))))
    dB = torch.mean(torch.abs(torch.log(torch.abs( B_guess / B_target))))
    #pdb.set_trace()
    return dT+dE+dB

def error_real_imag(guess,target):

    L1  = F.l1_loss(guess.real, target.real)
    L1 += F.l1_loss(guess.imag, target.imag)
    return L1

def gradient_loss(pred, target):
    """
    Computes a gradient (edge-aware) loss between pred and target.
    Both tensors should be [B, C, H, W].
    Returns a scalar loss.
    """
    # Compute gradients in x and y direction
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = F.l1_loss(pred_dx, target_dx)
    loss_y = F.l1_loss(pred_dy, target_dy)

    return loss_x + loss_y

def my_pearsonr(x, y, eps=1e-8):
    """
    Compute Pearson correlation coefficient between two tensors x and y.
    x and y must have the same shape.
    """
    x_mean = x.mean()
    y_mean = y.mean()

    xm = x - x_mean
    ym = y - y_mean

    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2) + eps)

    return r_num / r_den

   
import torch
import torch.nn.functional as F

def pearson_loss(pred, target, eps=1e-8):
    """
    pred, target: [B, 1, H, W]
    Computes 1 - Pearson correlation (mean over batch)
    """
    B = pred.shape[0]
    pred = pred.reshape(B, -1)
    target = target.reshape(B, -1)

    # zero-mean + variance normalize
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    pred = pred / (pred.norm(dim=1, keepdim=True) + eps)
    target = target / (target.norm(dim=1, keepdim=True) + eps)

    r = F.cosine_similarity(pred, target, dim=1)  # [B]
    return 1 - r.mean()

def ssim_loss(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Compute SSIM loss: 1 - SSIM (so that lower is better).
    pred, target: [B, C, H, W]
    """
    # Gaussian kernel for local statistics
    def gaussian_window(window_size, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g

    device = pred.device
    channel = pred.size(1)
    window = gaussian_window(window_size).to(device)
    window_2d = (window[:, None] * window[None, :]).unsqueeze(0).unsqueeze(0)
    window_2d = window_2d.repeat(channel, 1, 1, 1)

    mu_pred = F.conv2d(pred, window_2d, padding=window_size//2, groups=channel)
    mu_target = F.conv2d(target, window_2d, padding=window_size//2, groups=channel)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window_2d, padding=window_size//2, groups=channel) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window_2d, padding=window_size//2, groups=channel) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window_2d, padding=window_size//2, groups=channel) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return 1 - ssim_map.mean()  # SSIM loss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Positional encoding (2D sin-cos) ----
class PositionalEncoding2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "positional dim must be divisible by 4"

    def forward(self, h, w, device):
        d = self.dim // 4
        y = torch.arange(h, device=device).float()
        x = torch.arange(w, device=device).float()
        omega = torch.pow(10000, torch.arange(d, device=device).float() / d)

        y = y[:, None] / omega[None, :]
        x = x[:, None] / omega[None, :]

        pe_y = torch.cat([torch.sin(y), torch.cos(y)], dim=1)  # [H, 2d]
        pe_x = torch.cat([torch.sin(x), torch.cos(x)], dim=1)  # [W, 2d]

        pe = torch.zeros(h, w, self.dim, device=device)
        pe[:, :, 0:2*d] = pe_y[:, None, :]
        pe[:, :, 2*d:4*d] = pe_x[None, :, :]
        return pe  # [H, W, dim]

# ---- Patch embedding ----
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=256, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> tokens [B, N, D]
        x = self.proj(x)  # [B, D, H/p, W/p]
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, D], N=Hp*Wp
        return x, (Hp, Wp)

# ---- Transformer encoder block ----
class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_model),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # x: [B, N, D]
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

# ---- Optional pixel-level self-attention (proper projection around MHA) ----
class PixelSelfAttention(nn.Module):
    def __init__(self, in_ch=3, embed_dim=128, nhead=4):
        super().__init__()
        self.in_proj  = nn.Conv2d(in_ch, embed_dim, 1)
        self.norm     = nn.LayerNorm(embed_dim)
        self.attn     = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.out_proj = nn.Conv2d(embed_dim, in_ch, 1)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        z = self.in_proj(x)              # [B, D, H, W]
        z = z.flatten(2).transpose(1, 2) # [B, HW, D]
        z = self.attn(self.norm(z), self.norm(z), self.norm(z), need_weights=False)[0]
        z = z.transpose(1, 2).view(B, -1, H, W)  # [B, D, H, W]
        return self.out_proj(z)          # [B, C, H, W]

# ---- ViT model that matches your main_net interface ----
class main_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=3,
                 img_size=64, patch_size=4,
                 embed_dim=256, depth=8, num_heads=4, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0,
                 use_fc_bottleneck=False, fc_hidden=512, fc_spatial=4,  # kept for API compat; unused
                 rotation_prob=0,
                 use_cross_attention=False, attn_heads=1,               # kept for API compat
                 epochs=200, pool_type='max',
                 err_L1=1, err_Multi=1, err_Pear=1, err_SSIM=1, err_Grad=1, err_Power=1, err_Bisp=0, err_Cross=1,
                 suffix='', dropout_1=0, dropout_2=0, dropout_3=0):

        super().__init__()
        # store/forward needed flags
        self.rotation_prob = rotation_prob
        self.err_L1=err_L1; self.err_Multi=err_Multi; self.err_Pear=err_Pear
        self.err_SSIM=err_SSIM; self.err_Grad=err_Grad; self.err_Power=err_Power
        self.err_Bisp=err_Bisp; self.err_Cross=err_Cross

        # patch & pos
        self.patch_size = patch_size
        self.embed = PatchEmbed(in_chans=in_channels, embed_dim=embed_dim, patch_size=patch_size)
        self.pos2d = PositionalEncoding2D(embed_dim)

        # transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=embed_dim, nhead=num_heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # decode back to image
        # unpatchify path: [B, N, D] -> [B, D, Hp, Wp] -> upsample to [B, D/2, 2Hp, 2Wp] ... -> [B, out, H, W]
        dec_ch = embed_dim
        self.unproj = nn.Identity()

        self.up1 = nn.ConvTranspose2d(dec_ch, dec_ch//2, kernel_size=2, stride=2)  # x2
        self.up2 = nn.ConvTranspose2d(dec_ch//2, dec_ch//4, kernel_size=2, stride=2)  # x4 total
        self.to_out = nn.Conv2d(dec_ch//4, out_channels, kernel_size=3, padding=1)

        # optional pixel attention after decoding (correctly projected)
        self.pixel_attn = PixelSelfAttention(in_ch=out_channels, embed_dim=128, nhead=attn_heads) if use_cross_attention else None

        # multi-scale heads (derive from main)
        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))

    def forward(self, x):
        # x: [B, 1, H, W] or [B, H, W]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "H and W must be divisible by patch_size"

        tokens, (Hp, Wp) = self.embed(x)              # [B, N, D], N=Hp*Wp
        # add 2D pos enc
        pe = self.pos2d(Hp, Wp, x.device).view(1, Hp*Wp, -1)  # [1, N, D]
        tokens = tokens + pe

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)                    # [B, N, D]

        # unpatchify
        z = tokens.transpose(1, 2).view(B, -1, Hp, Wp)   # [B, D, Hp, Wp]
        z = self.up1(z)                                  # [B, D/2, 2Hp, 2Wp]
        z = F.gelu(z)
        z = self.up2(z)                                  # [B, D/4, 4Hp, 4Wp] == original if patch_size=8 and Hp=H/8
        z = F.gelu(z)
        out_main = self.to_out(z)                        # [B, out, H, W]

        if self.pixel_attn is not None:
            out_main = self.pixel_attn(out_main)         # proper attention over pixels

        # multi-scale heads compatible with your criterion
        out_d2 = F.avg_pool2d(out_main, kernel_size=2, stride=2)  # 1/2
        out_d3 = F.avg_pool2d(out_main, kernel_size=4, stride=4)  # 1/4
        out_d4 = F.avg_pool2d(out_main, kernel_size=8, stride=8)  # 1/8

        return out_main, out_d2, out_d3, out_d4

    # ---- reuse your existing loss code by delegation ----
    def criterion1(self, preds, target):
        # We'll call back into your existing helpers if they’re in scope.
        out_main, out_d2, out_d3, out_d4 = preds
        all_loss = {}

        if self.err_L1>0:
            all_loss['L1_0'] = self.err_L1 * F.l1_loss(out_main, target)

        if self.err_Multi>0:
            t_d2 = F.interpolate(target, size=out_d2.shape[-2:], mode="bilinear", align_corners=False)
            t_d3 = F.interpolate(target, size=out_d3.shape[-2:], mode="bilinear", align_corners=False)
            t_d4 = F.interpolate(target, size=out_d4.shape[-2:], mode="bilinear", align_corners=False)
            all_loss['L1_Multi'] = self.err_Multi * (
                F.l1_loss(out_d2, t_d2) + F.l1_loss(out_d3, t_d3) + F.l1_loss(out_d4, t_d4)
            )

        if self.err_SSIM>0:
            all_loss['SSIM'] = self.err_SSIM * (
                ssim_loss(out_main[:,0:1], target[:,0:1]) +
                ssim_loss(out_main[:,1:2], target[:,1:2]) +
                ssim_loss(out_main[:,2:3], target[:,2:3])
            ) / 3.0

        if self.err_Grad>0:
            all_loss['Grad'] = self.err_Grad * (
                gradient_loss(out_main[:,0:1], target[:,0:1]) +
                gradient_loss(out_main[:,1:2], target[:,1:2]) +
                gradient_loss(out_main[:,2:3], target[:,2:3])
            ) / 3.0

        if self.err_Pear>0:
            all_loss['Pear'] = self.err_Pear * (
                pearson_loss(out_main[:,0:1], target[:,0:1]) +
                pearson_loss(out_main[:,1:2], target[:,1:2]) +
                pearson_loss(out_main[:,2:3], target[:,2:3])
            ) / 3.0

        if self.err_Power>0:
            all_loss['Power'] = self.err_Power * power_spectra_crit(out_main, target)

        if self.err_Cross>0:
            all_loss['Cross'] = self.err_Cross * cross_spectra_crit(out_main, target)

        if self.err_Bisp>0:
            all_loss['Bisp'] = self.err_Bisp * bispectrum_crit(out_main, target)

        return all_loss

    def criterion(self, preds, target):
        losses = self.criterion1(preds, target)
        return sum(losses.values())

