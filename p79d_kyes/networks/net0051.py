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
from torch.utils.tensorboard import SummaryWriter

idd = 51
what = "50.  Plus norm"

#fname = "clm_take3_L=4.h5"
fname = 'p79d_subsets_S32_N5.h5'
fname = 'p79d_subsets_S128_N5.h5'
#ntrain = 400
#ntrain = 500
ntrain = 2000
#ntrain = 1000
#ntrain = 600
#ntrain = 4
#nvalid=3
nvalid=400
downsample = True
norm = True
def load_data():

    all_data= loader.loader(fname,ntrain=ntrain, nvalid=nvalid)
    return all_data

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)



def thisnet():


    model = main_net(base_channels=32, fc_spatial=4, use_fc_bottleneck=False)

    model.apply(init_weights)


    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    if 0:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    return model

def train(model,all_data):
    epochs  = 500
    lr = 1e-3
    #lr = 1e-4
    batch_size=64
    lr_schedule=[100]
    trainer(model,all_data,epochs=epochs,lr=lr,batch_size=batch_size, weight_decay=1e-3, lr_schedule=lr_schedule)

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
import torch
from torch.utils.data import Dataset

class DatasetNorm(Dataset):
    def __init__(self, X, mean_x=None, std_x=None,
                       mean_y=None, std_y=None,
                       compute_stats=False):
        if downsample:
            self.all_data=downsample_avg(X,32)
        else:
            self.all_data=X

        # preserve channel dimensions
        self.X = self.all_data[:, 0:1, :, :]   # [B, 1, H, W]
        self.y = self.all_data[:, 1:, :, :]    # [B, C, H, W]

        if compute_stats:
            # compute stats across training set
            dims = list(range(self.X.ndim))
            dims.remove(1)  # keep channel dimension
            self.mean_x = self.X.mean(dim=dims, keepdim=True)
            self.std_x  = self.X.std(dim=dims, keepdim=True, unbiased=False) + 1e-8

            self.mean_y = self.y.mean(dim=dims, keepdim=True)
            self.std_y  = self.y.std(dim=dims, keepdim=True, unbiased=False) + 1e-8
        else:
            assert mean_x is not None and std_x is not None
            assert mean_y is not None and std_y is not None
            self.mean_x, self.std_x = mean_x, std_x
            self.mean_y, self.std_y = mean_y, std_y

        # normalize
        self.X_norm = (self.X - self.mean_x) / self.std_x
        self.y_norm = (self.y - self.mean_y) / self.std_y

    def __getitem__(self, idx):
        return self.X_norm[idx], self.y_norm[idx]

    def __len__(self):
        return len(self.X)

    def unnormalize_y(self, y_norm):
        return y_norm * self.std_y + self.mean_y

    def get_stats(self):
        return self.mean_x, self.std_x, self.mean_y, self.std_y

class SphericalDataset(Dataset):
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

    if norm:
        ds_train = DatasetNorm(all_data['train'], compute_stats=True)
        ds_val   = DatasetNorm(all_data['valid'], mean_x=ds_train.mean_x, std_x=ds_train.std_x, mean_y=ds_train.mean_y, std_y=ds_train.std_y)
    else:
        ds_train = SphericalDataset(all_data['train'])
        ds_val   = SphericalDataset(all_data['valid'])
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(ds_val,   batch_size=max(64, batch_size), shuffle=False, drop_last=False)

    model = model.to(device)

    writer = SummaryWriter(log_dir="board/run_net%d/net_experiment"%idd)

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
        for step, (xb, yb) in enumerate(train_loader):
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
            global_step = epoch * len(train_loader) + step
            writer.add_scalar("Loss/train", loss.item(), global_step)

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
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f"Weights/{name}", param.data.cpu().numpy(), epoch)
                if param.grad is not None:
                    writer.add_histogram(f"Grads/{name}", param.grad.cpu().numpy(), epoch)

        

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


def error_real_imag(guess,target):

    L1  = F.l1_loss(guess.real, target.real)
    L1 += F.l1_loss(guess.imag, target.imag)
    return L1

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # Pre-activation BN + conv
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

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

        # Pre-activation residual block
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.dropout2d(out, p=self.dropout, training=self.training)

        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        # Optional second dropout
        # out = F.dropout2d(out, p=self.dropout, training=self.training)

        # SE attention
        w = self.global_pool(out).view(out.size(0), -1)
        w = F.relu(self.fc1(w))
        w = F.dropout(w, p=self.dropout, training=self.training)  # channel dropout in SE
        w = torch.sigmoid(self.fc2(w)).view(out.size(0), out.size(1), 1, 1)
        out = out * w

        # Skip connection
        if self.proj is not None:
            identity = self.proj(identity)
        out += identity

        return out

# ---------------- Main Net ----------------
class main_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_channels=32,
                 use_fc_bottleneck=False, fc_hidden=512, fc_spatial=4, dropout=0.1):
        super().__init__()
        self.use_fc_bottleneck = use_fc_bottleneck
        self.dropout = dropout

        # Encoder
        self.enc1 = ResidualBlockSE(in_channels, base_channels, dropout=dropout)
        self.enc2 = ResidualBlockSE(base_channels, base_channels*2, dropout=dropout)
        self.enc3 = ResidualBlockSE(base_channels*2, base_channels*4, dropout=dropout)
        self.enc4 = ResidualBlockSE(base_channels*4, base_channels*8, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

        # Optional FC bottleneck
        if use_fc_bottleneck:
            self.fc_spatial = fc_spatial
            self.fc1 = nn.Linear(base_channels*8*fc_spatial*fc_spatial, fc_hidden)
            self.fc2 = nn.Linear(fc_hidden, base_channels*8*fc_spatial*fc_spatial)

        # Learned upsampling via ConvTranspose2d
        self.up4 = nn.ConvTranspose2d(base_channels*8, base_channels*8, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels*2, kernel_size=2, stride=2)

        # Decoder with skip connections (no dropout here)
        self.dec4 = ResidualBlockSE(base_channels*8 + base_channels*4, base_channels*4, dropout=0.0)
        self.dec3 = ResidualBlockSE(base_channels*4 + base_channels*2, base_channels*2, dropout=0.0)
        self.dec2 = ResidualBlockSE(base_channels*2 + base_channels, base_channels, dropout=0.0)
        self.dec1 = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x):
        if x.ndim == 3: x = x.unsqueeze(1)

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
            z = F.dropout(z, p=self.dropout, training=self.training)
            z = F.relu(self.fc2(z))
            z = F.dropout(z, p=self.dropout, training=self.training)
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

        out = self.dec1(d2)
        return out

    def criterion(self, guess, target):
        return F.l1_loss(guess, target)

