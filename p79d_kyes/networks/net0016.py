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


idd = 16
what = "net 15 with less dil on the decoder"

#fname = "clm_take3_L=4.h5"
fname = 'p79d_subsets_S32_N5.h5'
fname = 'p79d_subsets_S128_N5.h5'
#ntrain = 400
ntrain = 4
#ntrain = 20
#ntrain = 600
#nvalid=3
nvalid=4
def load_data():

    all_data= loader.loader(fname,ntrain=ntrain, nvalid=nvalid)
    return all_data

def thisnet():

    model = main_net(hidden_dim=1024, base_channels=64, fc_spatial=8)

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model

def train(model,all_data):
    epochs  = 200
    lr = 1e-3
    #lr = 1e-4
    batch_size=10 
    lr_schedule=[100]
    trainer(model,all_data,epochs=epochs,lr=lr,batch_size=batch_size, weight_decay=0, lr_schedule=lr_schedule)

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

        model.train_curve = torch.tensor(train_curve)
        model.val_curve = torch.tensor(val_curve)

        print(f"[{epoch:3d}/{epochs}] net{idd:d}  train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {lr:.2e} | bad {bad_epochs:02d} | ETA {eta}")
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class main_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_channels=32, hidden_dim=512, fc_spatial=4):
        super().__init__()
        self.fc_spatial = fc_spatial  # spatial size for FC bottleneck

        # ----- Encoder -----
        self.enc1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, dilation=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=2, dilation=2)
        self.enc3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=4, dilation=4)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # ----- Fully connected bottleneck -----
        self.fc1 = nn.Linear(base_channels*4 * fc_spatial * fc_spatial, hidden_dim)
        #self.fc1a = nn.Linear(hidden_dim, 2*hidden_dim)
        #self.fc1b = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, base_channels*4 * fc_spatial * fc_spatial)

        # ----- Decoder -----
        # two-step decoding per stage for refinement
        self.dec3 = nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1, dilation=1)
        self.dec3a = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1)

        self.dec2 = nn.Conv2d(base_channels*4, base_channels, kernel_size=3, padding=1, dilation=1)
        self.dec2a = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # final layer takes skip from e1 and original input
        self.dec1 = nn.Conv2d(base_channels*2 + in_channels, out_channels, kernel_size=3, padding=1, dilation=1)

    def forward(self, x):
        # Ensure input has channel dimension
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [B, 1, H, W]

        # ----- Encoder -----
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(self.pool(e1)))
        e3 = F.relu(self.enc3(self.pool(e2)))

        # ----- FC bottleneck -----
        B, C, H, W = e3.shape
        z = F.adaptive_avg_pool2d(e3, (self.fc_spatial, self.fc_spatial))  # [B, C, fc_spatial, fc_spatial]
        z = z.view(B, -1)                                                  # [B, C*fc_spatial*fc_spatial]
        z = F.relu(self.fc1(z))
        #z = F.relu(self.fc1a(z))
        #z = F.relu(self.fc1b(z))
        z = F.relu(self.fc2(z))
        z = z.view(B, C, self.fc_spatial, self.fc_spatial)
        z = F.interpolate(z, size=(H, W), mode='bilinear', align_corners=False)

        # add residual from encoder (keeps detail)
        z = z + e3

        # ----- Decoder with skip connections -----
        d3 = F.relu(self.dec3(self.up(z)))
        d3 = F.relu(self.dec3a(d3))
        d3 = torch.cat([d3, e2], dim=1)

        d2 = F.relu(self.dec2(self.up(d3)))
        d2 = F.relu(self.dec2a(d2))
        d2 = torch.cat([d2, e1], dim=1)

        out = self.dec1(torch.cat([d2, x], dim=1))  # include raw input skip
        return out

    def criterion(self, guess, target):
        #return F.mse_loss(guess, target)  # MSE for sharper detail
        return F.l1_loss(guess, target)  # MSE for sharper detail


