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


idd = 1000
what = "302.  With one hot errors"

fname_train = "p79d_subsets_S256_N5_xyz_down_12823456_first.h5"
fname_valid = "p79d_subsets_S256_N5_xyz_down_12823456_second.h5"
#ntrain = 2000
#ntrain = 1000 #ntrain = 600
ntrain = 5
#ntrain = 3000
#nvalid=3
#ntrain = 10
nvalid=30
downsample = 64
#device = device or ("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
#epochs  = 20
epochs = 2000
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
# Dataset with rotation
# ---------------------------
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
    rng_min = all_data['train'][:,1:,:,:].min()
    rng_max = all_data['train'][:,1:,:,:].max()
    model.range=[rng_min,rng_max]


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    print("Total Steps", total_steps)
    if 0:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_schedule, #[100,300,600],  # change after N and N+M steps
            gamma=0.1             # multiply by gamma each time
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

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
                out_prob = model(xb)
                if verbose:
                    print("  crit")
                loss = model.criterion(out_prob,yb)

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
import torch
import torch.nn as nn
import torch.nn.functional as F


def meanie(model, out_prob):
    # Convert to probabilities
    prob = F.softmax(out_prob, dim=2)  # [B, 2, num_bins, H, W]

    # Expected value and variance per pixel
    bins = torch.linspace(model.range[0], model.range[1],
                          model.num_bins, device=prob.device)
    bins = bins.view(1, 1, -1, 1, 1)  # reshape for broadcasting

    mean = (prob * bins).sum(dim=2)  # [B, 2, H, W]
    var = (prob * (bins - mean.unsqueeze(2)) ** 2).sum(dim=2)  # [B, 2, H, W]
    return mean, var


class main_net(nn.Module):
    def __init__(self, num_bins=32, rng=(-1, 1)):
        super().__init__()
        self.num_bins = num_bins
        self.range = rng
        self.rotation_prob=0.0

        # --- Encoder ---
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        # --- Bottleneck ---
        self.bottleneck = nn.Conv2d(64, 64, 3, padding=1)

        # --- Decoder ---
        self.dec1 = nn.ConvTranspose2d(64, 32, 3, stride=2,
                                       padding=1, output_padding=1)
        self.dec2 = nn.Conv2d(32, 32, 3, padding=1)

        # --- Head: predicts logits for two distributions ---
        self.out_prob = nn.Conv2d(32, 2 * num_bins, 1)
        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))

    # ------------------------
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.bottleneck(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))

        out_prob = self.out_prob(x)  # [B, 2*num_bins, H, W]
        out_prob = out_prob.view(x.shape[0], 2, self.num_bins,
                                 x.shape[2], x.shape[3])
        return out_prob

    # ------------------------
    def criterion(self, out_prob, target):
        target = target[:,1:,:,:]
        """Total loss = CE(one-hot) + L1(mean)"""
        return self.criterion_hot(out_prob, target) + self.criterion_mean(out_prob, target)

    def criterion_mean(self, out_prob, target):
        """Compute L1 between target and mean of predicted distribution."""
        mean, _ = meanie(self, out_prob)
        l1 = F.l1_loss(mean, target)
        return l1

    # ------------------------
    def criterion_hot(self, out_prob, target):
        """
        out_prob: [B, 2, num_bins, H, W]
        target:   [B, 2, H, W]
        """
        B, two, num_bins, H, W = out_prob.shape
        bins = torch.linspace(self.range[0], self.range[1],
                              self.num_bins + 1, device=target.device)

        # Convert continuous targets to integer bin indices
        target_E = torch.bucketize(target[:, 0, ...], bins) - 1
        target_B = torch.bucketize(target[:, 1, ...], bins) - 1
        target_E = target_E.clamp(0, num_bins - 1).long()
        target_B = target_B.clamp(0, num_bins - 1).long()

        # Combine both maps into one batch
        logits = torch.cat([out_prob[:, 0, ...], out_prob[:, 1, ...]], dim=0)  # [2B, num_bins, H, W]
        target_bins = torch.cat([target_E, target_B], dim=0)                   # [2B, H, W]

        # Cross-entropy on raw logits (softmax handled inside)
        loss = F.cross_entropy(logits, target_bins)
        return loss

