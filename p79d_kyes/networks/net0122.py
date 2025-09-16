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


idd = 122
what = "118 y only, learned weights"

fname_train = "p79d_subsets_S512_N5_xyz_down_128_2356_y.h5"
fname_valid = "p79d_subsets_S512_N5_xyz_down_128_4y.h5"
ntrain = 2000
#ntrain = 600
#nvalid=3
#ntrain = 10
nvalid=30
downsample = 64
#device = device or ("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs  = 300
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

    model = main_net(base_channels=32, fc_spatial=4, use_fc_bottleneck=fc_bottleneck)

    #model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

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

class main_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_channels=32,
                 use_fc_bottleneck=False, fc_hidden=512, fc_spatial=4):
        super().__init__()
        self.use_fc_bottleneck = use_fc_bottleneck

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

        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))
        self.loss_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32))

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



        return out_main, out_d2, out_d3, out_d4

    def criterion(self, preds, target):
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
        #total_loss = loss_main + 0.5 * loss_d2 + 0.25 * loss_d3 + 0.125 * loss_d4
        losses = torch.stack([
            F.l1_loss(out_main, target),
            F.l1_loss(out_d2, t_d2),
            F.l1_loss(out_d3, t_d3),
            F.l1_loss(out_d4, t_d4)
        ])

        # Ensure weights are positive (ReLU or softplus)
        weights = F.softplus(self.loss_weights)
        weights = weights / weights.sum()  # normalize to sum=1 (optional)

        weights=weights.to(device)
        losses=losses.to(device)
        total_loss = torch.sum(weights * losses)
        return total_loss
