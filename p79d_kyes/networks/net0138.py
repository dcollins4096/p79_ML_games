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


idd = 138
what = "137 ps; a=1,1 b=2,2 (pearson, power); c=log abs; d=all data, capacity 32, 16>16,4"

fname_train = "p79d_subsets_S512_N5_xyz_down_128_2356.h5"
fname_valid = "p79d_subsets_S512_N5_xyz_down_128_4.h5"
#ntrain = 2000
#ntrain = 1000 #ntrain = 600
ntrain = 3000
#nvalid=3
#ntrain = 10
nvalid=30
downsample = False
#device = device or ("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs  = 200
#epochs = 20
lr = 1e-3
#lr = 1e-4
batch_size=10 
lr_schedule=[100]
weight_decay = 1e-3
fc_bottleneck=True
def load_data():

    print('read the data')
    train= loader.loader(fname_train,ntrain=ntrain, nvalid=nvalid)
    valid= loader.loader(fname_valid,ntrain=1, nvalid=nvalid)
    all_data={'train':train['train'],'valid':valid['valid']}
    print('done')
    return all_data

def thisnet():

    model = main_net(base_channels=32,fc_hidden=512 , fc_spatial=16, use_fc_bottleneck=fc_bottleneck, out_channels=3, use_cross_attention=False, attn_heads=1)

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
class SphericalDataset(Dataset):
    def __init__(self, all_data):
        if downsample:
            self.all_data=downsample_avg(all_data,downsample)
        else:
            self.all_data=all_data
    def __len__(self):
        return self.all_data.size(0)

    def __getitem__(self, idx):
        #return self.data[idx], self.targets[idx]
        return self.all_data[idx][0], self.all_data[idx]

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

    ds_train = SphericalDataset(all_data['train'])
    ds_val   = SphericalDataset(all_data['valid'])
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

    for epoch in range(1, epochs+1):
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

def error_real_imag(guess,target):

    L1  = F.l1_loss(guess.real, target.real)
    L1 += F.l1_loss(guess.imag, target.imag)
    return L1

# ---------------- Residual SE Block ----------------
class ResidualBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection if channel mismatch
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1)

        # SE attention
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // reduction)
        self.fc2 = nn.Linear(out_channels // reduction, out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # SE attention
        w = self.global_pool(out).view(out.size(0), -1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).view(out.size(0), out.size(1), 1, 1)
        out = out * w

        # Skip connection
        if self.proj is not None:
            identity = self.proj(identity)
        out += identity
        return F.relu(out)

# ---------------- Main Net ----------------
import torch
import torch.nn.functional as F

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

class CrossAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # -> [B, HW, C]
        x_attn, _ = self.attn(x_flat, x_flat, x_flat) # self-attention
        return x_attn.transpose(1, 2).view(B, C, H, W)


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

class main_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_channels=32,
                 use_fc_bottleneck=False, fc_hidden=512, fc_spatial=4,
                 use_cross_attention=True, attn_heads=1, epochs=300):
        super().__init__()
        self.use_fc_bottleneck = use_fc_bottleneck
        self.use_cross_attention = use_cross_attention

        # Encoder
        self.enc1 = ResidualBlockSE(in_channels, base_channels)
        self.enc2 = ResidualBlockSE(base_channels, base_channels*2)
        self.enc3 = ResidualBlockSE(base_channels*2, base_channels*4)
        self.enc4 = ResidualBlockSE(base_channels*4, base_channels*8)
        self.pool = nn.MaxPool2d(2)

        # Optional FC bottleneck
        if use_fc_bottleneck:
            self.fc_spatial = fc_spatial
            self.fc1 = nn.Linear(base_channels*8*fc_spatial*fc_spatial, fc_hidden)
            self.fc2 = nn.Linear(fc_hidden, base_channels*8*fc_spatial*fc_spatial)

        # Learned upsampling via ConvTranspose2d
        self.up4 = nn.ConvTranspose2d(base_channels*8, base_channels*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Decoder with skip connections
        self.dec4 = ResidualBlockSE(base_channels*8 + base_channels*4, base_channels*4)
        self.dec3 = ResidualBlockSE(base_channels*4 + base_channels*2, base_channels*2)
        self.dec2 = ResidualBlockSE(base_channels*2 + base_channels, base_channels)
        self.dec1 = nn.Conv2d(base_channels, out_channels, 3, padding=1)

        # --- Multi-scale output heads ---
        self.out_d4 = nn.Conv2d(base_channels*4, out_channels, 3, padding=1)
        self.out_d3 = nn.Conv2d(base_channels*2, out_channels, 3, padding=1)
        self.out_d2 = nn.Conv2d(base_channels,   out_channels, 3, padding=1)

        # Optional cross-attention
        if use_cross_attention:
            self.cross_attn = CrossAttention(out_channels, num_heads=attn_heads)

        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))

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
            z = F.relu(self.fc2(z))
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
        out_d4 = self.out_d4(d4)
        out_d3 = self.out_d3(d3)
        out_d2 = self.out_d2(d2)

        if self.use_cross_attention:
            out_main = self.cross_attn(out_main)

        return out_main, out_d2, out_d3, out_d4

    def criterion1(self, preds, target):
        """
        preds: tuple of (out_main, out_d2, out_d3, out_d4)
        target: [B, C, H, W] ground truth
        """
        out_main, out_d2, out_d3, out_d4 = preds

        # Downsample target to match each prediction
        t_d2 = F.interpolate(target, size=out_d2.shape[-2:], mode="bilinear", align_corners=False)
        t_d3 = F.interpolate(target, size=out_d3.shape[-2:], mode="bilinear", align_corners=False)
        t_d4 = F.interpolate(target, size=out_d4.shape[-2:], mode="bilinear", align_corners=False)

        loss_main = F.l1_loss(out_main, target)
        loss_d2   = F.l1_loss(out_d2, t_d2)
        loss_d3   = F.l1_loss(out_d3, t_d3)
        loss_d4   = F.l1_loss(out_d4, t_d4)

        # Weighted sum (more weight on full-res output)
        ssim_t  = ssim_loss(out_main[:,0:1,:,:], target[:,0:1,:,:])
        ssim_e  = ssim_loss(out_main[:,1:2,:,:], target[:,1:2,:,:])
        ssim_b  = ssim_loss(out_main[:,2:3,:,:], target[:,2:3,:,:])
        grad_t  = gradient_loss(out_main[:,0:1,:,:], target[:,0:1,:,:])
        grad_e  = gradient_loss(out_main[:,1:2,:,:], target[:,1:2,:,:])
        grad_b  = gradient_loss(out_main[:,2:3,:,:], target[:,2:3,:,:])
        pear_t  = 1-my_pearsonr(out_main[:,0:1,:,:].flatten(), target[:,0:1,:,:].flatten())
        pear_e  = 1-my_pearsonr(out_main[:,1:2,:,:].flatten(), target[:,1:2,:,:].flatten())
        pear_b  = 1-my_pearsonr(out_main[:,2:3,:,:].flatten(), target[:,2:3,:,:].flatten())
        lambda_pear = 1.0*(pear_e+pear_b+pear_t)

        lambda_ssim = 1.0*(ssim_e+ssim_b+ssim_t)
        lambda_grad = 1.0*(grad_e+grad_b+grad_t)
        lambda_power = power_spectra_crit(out_main, target)


        stuff=[loss_main, loss_d2, loss_d3, loss_d4, lambda_ssim, lambda_grad, lambda_pear, lambda_power]
        out = torch.stack(stuff).to(device)
        return out

    def criterion(self, preds, target):
        losses = self.criterion1(preds,target)
        weights = torch.tensor([1,0.5,0.25,0.125,1,1,2,1]).to(device)
        return (losses*weights).sum()

