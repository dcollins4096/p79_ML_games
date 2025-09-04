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


idd = 37
what = "net 34, loss function games"
idd = 38
what = "net 34, try to learn"
idd = 39
what = "net 34, try to learn, L1 loss"

#fname = "clm_take3_L=4.h5"
fname = 'p79d_subsets_S32_N5.h5'
fname = 'p79d_subsets_S128_N5.h5'
#ntrain = 400
#ntrain = 4
#ntrain = 2000
#ntrain = 1000
#ntrain = 600
ntrain = 20
#nvalid=3
nvalid=4
downsample = True
def load_data():

    all_data= loader.loader(fname,ntrain=ntrain, nvalid=nvalid)
    return all_data

def thisnet():

    model = main_net(base_channels=32, fc_spatial=4, use_fc_bottleneck=False)

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model

def train(model,all_data):
    epochs  = 500
    lr = 1e-3
    #lr = 1e-4
    batch_size=10 
    lr_schedule=[100]
    trainer(model,all_data,epochs=epochs,lr=lr,batch_size=batch_size, weight_decay=0, lr_schedule=lr_schedule)

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
    """
    data:  (N, 3, T)  with rows [theta, phi, rm]
    label: (N, D_out)
    We normalize:
      - theta in [0,pi]  -> scaled to [-1,1] via (theta/pi)*2-1
      - phi   in [0,2pi] -> scaled to [-1,1] via (phi/pi)-1
      - rm    -> standardized by train mean/std (computed here across full tensor)
    """
    def __init__(self, all_data):
        if downsample:
            self.all_data=downsample_avg(all_data,32)
        else:
            self.all_data=all_data

        



    def __len__(self):
        return self.all_data.size(0)

    def __getitem__(self, idx):
        #return self.data[idx], self.targets[idx]
        return self.all_data[idx][0], self.all_data[idx][1:]

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
    device=None,
    lr_schedule=[900],
    plot_path=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
                vloss = model.criterion(preds, yb)
                vtotal += vloss.item() * xb.size(0)
            val_loss = vtotal / len(ds_val)
            val_curve.append(val_loss)

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

        model.train_curve = torch.tensor(train_curve)
        model.val_curve = torch.tensor(val_curve)

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
    plt.plot(model.train_curve, label="train")
    plt.plot(model.val_curve,   label="val")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/plots/errtime_net%04d"%(os.environ['HOME'], model.idd))

def fft_loss(pred, target, reduction="mean"):
    """
    pred, target: tensors of shape [B, C, H, W]
    Computes L1 difference in Fourier magnitude.
    """
    # Compute FFT
    pred_fft = torch.fft.fft2(pred, norm="ortho")
    target_fft = torch.fft.fft2(target, norm="ortho")

    # Magnitude spectra
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    # L1 difference
    loss = torch.abs(pred_mag - target_mag)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        # --- Residual path ---
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # --- Skip connection projection if channels mismatch ---
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1)

        # --- SE attention ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // reduction)
        self.fc2 = nn.Linear(out_channels // reduction, out_channels)

    def forward(self, x):
        identity = x

        # Residual path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Channel attention
        w = self.global_pool(out).view(out.size(0), -1)   # [B, C]
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))                    # [B, C]
        w = w.view(out.size(0), out.size(1), 1, 1)        # reshape for broadcast
        out = out * w                                     # scale channels

        # Skip connection
        if self.proj is not None:
            identity = self.proj(identity)

        out += identity
        return F.relu(out)



class main_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_channels=32,
                 use_fc_bottleneck=False, fc_hidden=512, fc_spatial=4):
        super().__init__()
        self.use_fc_bottleneck = use_fc_bottleneck

        # --- Encoder ---
        self.enc1 = ResidualBlock(in_channels, base_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels*2)
        self.enc3 = ResidualBlock(base_channels*2, base_channels*4)
        self.enc4 = ResidualBlock(base_channels*4, base_channels*8)
        self.pool = nn.MaxPool2d(2)

        # --- Optional Fully Connected Bottleneck ---
        if use_fc_bottleneck:
            self.fc_spatial = fc_spatial
            self.fc1 = nn.Linear(base_channels*8 * fc_spatial * fc_spatial, fc_hidden)
            self.fc2 = nn.Linear(fc_hidden, base_channels*8 * fc_spatial * fc_spatial)

        # --- Learned Upsample + Conv ---
        self.up4_conv = ResidualBlock(base_channels*8, base_channels*8)
        self.up3_conv = ResidualBlock(base_channels*4, base_channels*4)
        self.up2_conv = ResidualBlock(base_channels*2, base_channels*2)

        # --- Decoder ---
        self.dec4 = ResidualBlock(base_channels*8 + base_channels*4, base_channels*4)
        self.dec3 = ResidualBlock(base_channels*4 + base_channels*2, base_channels*2)
        self.dec2 = ResidualBlock(base_channels*2 + base_channels, base_channels)
        self.dec1 = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # --- Encoder ---
        e1 = self.enc1(x)                 # [B, C, H, W]
        e2 = self.enc2(self.pool(e1))     # [B, 2C, H/2, W/2]
        e3 = self.enc3(self.pool(e2))     # [B, 4C, H/4, W/4]
        e4 = self.enc4(self.pool(e3))     # [B, 8C, H/8, W/8]

        # --- Optional FC Bottleneck ---
        if self.use_fc_bottleneck:
            B, C, H, W = e4.shape
            z = F.adaptive_avg_pool2d(e4, (self.fc_spatial, self.fc_spatial))
            z = z.view(B, -1)
            z = F.relu(self.fc1(z))
            z = F.relu(self.fc2(z))
            z = z.view(B, C, self.fc_spatial, self.fc_spatial)
            e4 = F.interpolate(z, size=(H, W), mode='bilinear', align_corners=False)

        # --- Decoder ---
        d4 = F.interpolate(e4, scale_factor=2, mode='nearest')
        d4 = self.up4_conv(d4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = F.interpolate(d4, scale_factor=2, mode='nearest')
        d3 = self.up3_conv(d3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(d3, scale_factor=2, mode='nearest')
        d2 = self.up2_conv(d2)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        out = self.dec1(d2)
        return out

    def criterion(self, guess, target):
        L1 = F.l1_loss(guess, target)
        #FT = fft_loss(guess,target)

        return L1


