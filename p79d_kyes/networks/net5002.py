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


idd = 5002
what = "Fresh network, now with supervision"

#fname_train = "p79d_subsets_S256_N5_xyz_down_12823456_first.h5"
#fname_valid = "p79d_subsets_S256_N5_xyz_down_12823456_second.h5"
fname_train = "p79d_subsets_S512_N5_xyz__down_64T_first.h5"
fname_valid = "p79d_subsets_S512_N5_xyz__down_64T_second.h5"

fname_train = "p79d_subsets_S512_N3_xyz_T_first.h5"
fname_valid = "p79d_subsets_S512_N3_xyz_T_second.h5"

fname_train = "p79d_subsets_S128_N1_xyz_suite7vs_first.h5"
fname_valid = "p79d_subsets_S128_N1_xyz_suite7vs_second.h5"



#ntrain = 2000
#ntrain = 1000 #ntrain = 600
#ntrain = 20
ntrain = 1400
#nvalid=3
#ntrain = 10
nvalid=30
ntest = 5000
downsample = None
#device = device or ("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
#epochs  = 1e6
epochs = 30
lr = 0.5e-3
#lr = 1e-4
batch_size=64
lr_schedule=[1000]
weight_decay = 1e-2
fc_bottleneck=True
def load_data():

    print('read the data')
    train= loader.loader(fname_train,ntrain=ntrain, nvalid=nvalid)
    valid= loader.loader(fname_valid,ntrain=1, nvalid=nvalid)
    all_data={'train':train['train'],'valid':valid['valid'], 'test':valid['test'][:ntest], 'quantities':{}}
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
    def __init__(self, all_data, quan, rotation_prob = 0.0, rand=False):
        self.rotation_prob = rotation_prob
        self.quan=quan
        if downsample:
            self.all_data=downsample_avg(all_data,downsample)
        else:
            self.all_data=all_data
        self.rand=rand
    def __len__(self):
        return self.all_data.size(0)

    def __getitem__(self, idx):
        #return self.data[idx], self.targets[idx]
        H, W = self.all_data[0][0].shape
        dy = torch.randint(0, H, (1,)).item()
        dx = torch.randint(0, W, (1,)).item()
        theset= torch.roll(self.all_data[idx], shifts=(dy, dx), dims=(-2, -1))
        ms = self.quan['Ms_act'][idx]
        ma = self.quan['Ma_act'][idx]
        theset[2] = torch.sqrt(theset[2])
        return theset[0:3].to(device), torch.tensor([ms], dtype=torch.float32).to(device)

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

    ds_train = SphericalDataset(all_data['train'],all_data['quantities']['train'], rand=True)
    ds_val   = SphericalDataset(all_data['valid'],all_data['quantities']['valid'])
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(ds_val,   batch_size=max(64, batch_size), shuffle=False, drop_last=False)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    print("Total Steps", total_steps, "ntrain", min(ntrain,len(all_data['train'])), "epoch", epochs, "down", downsample)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_schedule, #[100,300,600],  # change after N and N+M steps
        gamma=0.1             # multiply by gamma each time
    )

    best_val = float("inf")
    best_state = None
    load_best = False
    patience = epochs
    bad_epochs = 0

    train_curve, val_curve = [], []
    t0 = time.time()
    verbose=False


    for epoch in range(1, epochs+1):
        model.train()
        if verbose:
            print("Epoch %d"%epoch)
        running = 0.0
        import tqdm
        for xb, yb in tqdm.tqdm(train_loader):
            xb = xb.to(device)
            #yb = yb.to(device)

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
                #yb = yb.to(device)
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

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}. Best val {best_val:.4f}.")
            break

    # restore best
    if best_state is not None and load_best:
        model.load_state_dict(best_state)
    return model

#main_net
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- SE residual block (similar spirit to your ResidualBlockSE) ----------
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc1 = nn.Linear(c, max(1, c // r))
        self.fc2 = nn.Linear(max(1, c // r), c)

    def forward(self, x):
        b, c, h, w = x.shape
        s = x.mean(dim=(2,3))                 # [B,C]
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s

class ResSE(nn.Module):
    def __init__(self, c_in, c_out, stride=1, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)
        self.se    = SEBlock(c_out)
        self.drop  = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.skip = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        h = F.silu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.silu(h + self.skip(x))


# ---------- Attention pooling (learned spatial weighting) ----------
class AttnPool2d(nn.Module):
    """
    Produces a pooled vector by learning spatial weights.
    Returns [B, C]
    """
    def __init__(self, c):
        super().__init__()
        self.score = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        logits = self.score(x).view(b, -1)               # [B, HW]
        wts = torch.softmax(logits, dim=1).view(b, 1, h*w)
        xflat = x.view(b, c, h*w)
        pooled = torch.sum(xflat * wts, dim=2)           # [B, C]
        return pooled


# ---------- Optional: simple radial power spectrum features ----------
def radial_psd_features(x, nbins=32, eps=1e-12):
    """
    x: [B, C, H, W] -> returns [B, C*nbins] of log radially averaged power
    Differentiable. Uses fixed binning on the fly.
    """
    B, C, H, W = x.shape
    X = torch.fft.rfftn(x, dim=(-2, -1))
    P = (X.real**2 + X.imag**2)  # [B,C,H,W//2+1]

    # build radius grid for rfft layout
    device = x.device
    ky = torch.fft.fftfreq(H, device=device).view(H, 1)           # [-.5,.5)
    kx = torch.fft.rfftfreq(W, device=device).view(1, W//2 + 1)   # [0,.5]
    rr = torch.sqrt(ky**2 + kx**2)                                # [H, W//2+1]
    rr = rr / (rr.max() + eps)

    # bin indices 0..nbins-1
    bins = torch.clamp((rr * nbins).long(), 0, nbins-1)           # [H, W//2+1]
    bins_flat = bins.view(-1)                                      # [HW']
    P_flat = P.view(B, C, -1)                                      # [B,C,HW']

    # accumulate sum and count per bin
    feats = []
    for k in range(nbins):
        mask = (bins_flat == k).float().view(1, 1, -1)            # [1,1,HW']
        num = (P_flat * mask).sum(dim=2)
        den = mask.sum(dim=2).clamp_min(1.0)
        avg = num / den
        feats.append(torch.log(avg + eps))                         # [B,C]
    feat = torch.stack(feats, dim=2)                               # [B,C,nbins]
    return feat.reshape(B, C*nbins)


# ---------- Main model: deep supervision + attention pooling + (optional) spectral features ----------
class main_net(nn.Module):
    def __init__(
        self,
        in_ch=3,
        base=32,
        drop=0.05,
        use_psd_features=True,
        psd_bins=24,
        deep_supervision=True,
        loss_w_deep=0.5,   # weight of auxiliary heads vs final head
    ):
        super().__init__()
        self.use_psd_features = use_psd_features
        self.psd_bins = psd_bins
        self.deep_supervision = deep_supervision
        self.loss_w_deep = loss_w_deep

        # Keep early spatial info (avoid aggressive stem)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU()
        )

        # Encoder stages (like your 4 enc blocks)
        self.e1 = nn.Sequential(ResSE(base, base, drop=drop), ResSE(base, base, drop=drop))
        self.e2 = nn.Sequential(ResSE(base, base*2, stride=2, drop=drop), ResSE(base*2, base*2, drop=drop))
        self.e3 = nn.Sequential(ResSE(base*2, base*4, stride=2, drop=drop), ResSE(base*4, base*4, drop=drop))
        self.e4 = nn.Sequential(ResSE(base*4, base*8, stride=2, drop=drop), ResSE(base*8, base*8, drop=drop))

        # Pooling modules per stage
        self.pool1 = AttnPool2d(base)
        self.pool2 = AttnPool2d(base*2)
        self.pool3 = AttnPool2d(base*4)
        self.pool4 = AttnPool2d(base*8)

        # Head dims: (mean+std+attn) = 3*C per stage
        def stage_vec_dim(c):
            d = 3*c
            if self.use_psd_features:
                d += c * self.psd_bins
            return d

        self.h1 = nn.Sequential(nn.Linear(stage_vec_dim(base), 256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256, 2))
        self.h2 = nn.Sequential(nn.Linear(stage_vec_dim(base*2), 256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256, 2))
        self.h3 = nn.Sequential(nn.Linear(stage_vec_dim(base*4), 256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256, 2))
        self.h4 = nn.Sequential(nn.Linear(stage_vec_dim(base*8), 512), nn.SiLU(), nn.Dropout(0.2), nn.Linear(512, 2))

        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))
        self.n_scalars = 2
        self.predict_mu_sigma = True
        self.predict_scalars = True
    def _stage_vec(self, x, attn_pool):
        # mean+std pooling
        mean = x.mean(dim=(2,3))
        std  = x.flatten(2).std(dim=2)
        attn = attn_pool(x)
        z = torch.cat([mean, std, attn], dim=1)
        if self.use_psd_features:
            psd = radial_psd_features(x, nbins=self.psd_bins)  # [B, C*nbins]
            z = torch.cat([z, psd], dim=1)
        return z

    def forward(self, x):
        x = self.stem(x)
        f1 = self.e1(x)        # [B, base, H, W]
        f2 = self.e2(f1)       # [B, 2b, H/2, W/2]
        f3 = self.e3(f2)       # [B, 4b, H/4, W/4]
        f4 = self.e4(f3)       # [B, 8b, H/8, W/8]

        z1 = self._stage_vec(f1, self.pool1)
        z2 = self._stage_vec(f2, self.pool2)
        z3 = self._stage_vec(f3, self.pool3)
        z4 = self._stage_vec(f4, self.pool4)

        o1 = self.h1(z1)  # [B,2] mean/logvar
        o2 = self.h2(z2)
        o3 = self.h3(z3)
        o4 = self.h4(z4)

        # return multiscale outputs (like your old multi-head behavior)
        return (o4, o3, o2, o1)

    @staticmethod
    def gaussian_nll(mean_logvar, target):
        mean, logvar = mean_logvar
        logvar = logvar.clamp(-10, 5)
        return 0.5 * (logvar + (target - mean)**2 * torch.exp(-logvar)).mean()

    def criterion(self, preds, target):
        """
        preds = (o4, o3, o2, o1) each [B,2]
        target expected shape [B] or [B,1]
        """
        if target.ndim == 2 and target.shape[1] == 1:
            target = target[:, 0]

        o4, o3, o2, o1 = preds
        # main loss uses deepest features
        L4 = self.gaussian_nll((o4[:,0], o4[:,1]), target)

        if not self.deep_supervision:
            return L4

        L3 = self.gaussian_nll((o3[:,0], o3[:,1]), target)
        L2 = self.gaussian_nll((o2[:,0], o2[:,1]), target)
        L1 = self.gaussian_nll((o1[:,0], o1[:,1]), target)
        Ldeep = (L3 + L2 + L1) / 3.0

        return (1.0 - self.loss_w_deep) * L4 + self.loss_w_deep * Ldeep

