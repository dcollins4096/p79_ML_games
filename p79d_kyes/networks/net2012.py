import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torchvision.transforms.functional as TF
import random
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


idd = 2012
what = "2008, try to improve"

#fname_train = "p79d_subsets_S256_N5_xyz_down_12823456_first.h5"
#fname_valid = "p79d_subsets_S256_N5_xyz_down_12823456_second.h5"
#fname_train = "p79d_subsets_S256_N5_xyzsuite4_first.h5"
#fname_valid = "p79d_subsets_S256_N5_xyzsuite4_second.h5"
fname_train = "p79d_subsets_S256_N5_xyz_down_64suite4_first.h5"
fname_valid = "p79d_subsets_S256_N5_xyz_down_64suite4_second.h5"
#ntrain = 2000
#ntrain = 1000 #ntrain = 600
#ntrain = 20
#ntrain = 3000
ntrain = 18000
#ntrain = 20
#nvalid=3
#ntrain = 10
nvalid=30
downsample = 64
#device = device or ("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs  = 20
#epochs = 200
#lr = 1e-3
lr = 1e-3
batch_size=10
lr_schedule=[400]
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

    model = main_net(base_channels=32,fc_hidden=512 , fc_spatial=4, use_fc_bottleneck=fc_bottleneck, out_channels=3, use_cross_attention=False, attn_heads=1)

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
# Dataset with input normalization
# ---------------------------
class SphericalDataset(Dataset):
    def __init__(self, all_data, rotation_prob = 0.0, downsample=downsample):
        self.rotation_prob = rotation_prob
        if downsample:
            self.all_data=downsample_avg(all_data,downsample)
        else:
            self.all_data=all_data
        #normalize E&B
        ebmean = self.all_data[:,1:,...].mean()
        ebstd  = self.all_data[:,1:,...].std()

        self.all_data[:,1:,...] = (self.all_data[:,1:,...]-ebmean)/ebstd
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

    ds_train = SphericalDataset(all_data['train'], rotation_prob=model.rotation_prob, downsample=downsample)
    ds_val   = SphericalDataset(all_data['valid'], rotation_prob=model.rotation_prob, downsample=downsample)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(ds_val,   batch_size=max(64, batch_size), shuffle=False, drop_last=False)

    model = model.to(device)
    rng_min = all_data['train'][:,1:,:,:].min()
    rng_max = all_data['train'][:,1:,:,:].max()
    model.range=torch.tensor([rng_min,rng_max], device=device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    print("Total Steps", total_steps)
    pdb.set_trace()
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


    import tqdm
    for epoch in (range(1, epochs+1)):
        if epoch > 50 and save_err_Cross>0:
            model.err_Cross = save_err_Cross
        model.train()
        if verbose:
            print("Epoch %d"%epoch)
        running = 0.0
        for xb, yb in tqdm.tqdm(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            if verbose:
                print("  model")
            if 1:
                preds = model(xb, return_features=True)
                if verbose:
                    print("  crit")

                #loss  = model.criterion(preds, yb[:,0:1,:,:])
                loss  = model.criterion(preds, yb, epoch=epoch)

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
                preds = model(xb, return_features=True)
                #vloss = model.criterion(preds, yb[:,0:1,:,:])
                vloss = model.criterion(preds, yb, epoch=epoch)
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

def power_spectrum_delta(guess,target):
    T_guess = torch_power.powerspectrum(guess)
    T_target = torch_power.powerspectrum(target)
    output = torch.mean( torch.abs(torch.log(T_guess.avgpower/(T_target.avgpower+1e-8))))
    return output

def power_spectra_crit(guess,target):
    err_T = power_spectrum_delta(guess[:,0:1,:,:], target[:,0:1,:,:])
    err_E = power_spectrum_delta(guess[:,1:2,:,:], target[:,1:2,:,:])
    err_B = 0
    if guess.shape[1]==3:
        err_B = power_spectrum_delta(guess[:,2:3,:,:], target[:,2:3,:,:])
    return err_T+err_E+err_B

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

# ======================================================
# Conditional Normalizing Flow Head for E/B prediction
# ======================================================
from nflows import flows, distributions, transforms

from nflows import flows, distributions, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EBFlowHead(nn.Module):
    def __init__(self, in_channels, context_dim=32, hidden_features=128,
                 num_layers=6, edge_width=8):
        super().__init__()
        self.edge_width = edge_width
        self.context_dim = context_dim

        # --- context compressor ---
        self.context_net = nn.Sequential(
            nn.Conv2d(in_channels + 2, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, context_dim, 3, padding=1),
            nn.GELU(),
            nn.InstanceNorm2d(context_dim, affine=False)
        )

        # --- spline coupling layers (manual context injection) ---
        def make_coupling(mask_pattern):
            class ContextNet(nn.Module):
                def __init__(self, in_features, out_features, context_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_features + context_dim, hidden_features),
                        nn.ReLU(),
                        nn.Linear(hidden_features, hidden_features),
                        nn.ReLU(),
                        nn.Linear(hidden_features, out_features),
                    )
                def forward(self, x, context=None):
                    if context is not None:
                        x = torch.cat([x, context], dim=1)
                    return self.net(x)

            def make_net(in_features, out_features):
                # returns a small module that concatenates context internally
                return ContextNet(in_features, out_features, self.context_dim)

            return transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=torch.tensor(mask_pattern, dtype=torch.bool),
                transform_net_create_fn=make_net,
                num_bins=8,
                tails='linear',
                tail_bound=3.0
            )

        chain = []
        masks = [[1, 0, 1], [0, 1, 0]] * (num_layers // 2 + 1)
        for i in range(num_layers):
            chain += [
                transforms.ActNorm(3),
                make_coupling(masks[i]),
                transforms.RandomPermutation(3),
            ]
        self.transform = transforms.CompositeTransform(chain)
        self.base = distributions.StandardNormal([3])
        self.flow = flows.Flow(self.transform, self.base)

    # --- cosine window helper ---
    def cosine_window(self, H, W, device):
        y = torch.linspace(0, math.pi, H, device=device)
        x = torch.linspace(0, math.pi, W, device=device)
        wy = 0.5 * (1 - torch.cos(y))
        wx = 0.5 * (1 - torch.cos(x))
        win = wy[:, None] * wx[None, :]
        Wmask = 1 - win
        return Wmask / Wmask.mean()

    def forward(self, features, target=None, mode="train"):

        B, C, H, W = features.shape
        device = features.device

        # Add coordinate channels
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )
        coords = torch.stack([xx, yy]).expand(B, -1, H, W)
        features = torch.cat([features, coords], dim=1)

        # Build context tensor
        context_feat = self.context_net(features)
        context = context_feat.permute(0, 2, 3, 1).reshape(-1, self.context_dim)

        if mode == "train":
            assert target is not None
            y = target.permute(0, 2, 3, 1).reshape(-1, 3)

            # Log-likelihood
            log_prob = self.flow.log_prob(y, context=context).view(B, H, W)
            Wmask = self.cosine_window(H, W, device)
            nll = -(log_prob * Wmask).mean()

            # Mode (z=0) sample
            z0 = torch.zeros_like(y)
            sample0, _ = self.flow._transform.inverse(z0, context=context)
            sample_res = sample0.view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

            # Windowed comparisons
            win = Wmask[None, None]
            sample_w = sample_res * win
            target_w = target * win
            ssim_term = ssim_loss(sample_w, target_w)
            pwr_term = power_spectra_crit(sample_w, target_w)

            tv = (
                (sample_res[:, :, :, 1:] - sample_res[:, :, :, :-1]).abs().mean()
                + (sample_res[:, :, 1:, :] - sample_res[:, :, :-1, :]).abs().mean()
            ) * 1e-5

            loss = nll + 0.1 * ssim_term + 0.05 * pwr_term + tv
            return loss

        elif mode == "sample":
            n = B * H * W
            z = self.base.sample(n)
            samples, _ = self.flow._transform.inverse(z, context=context)
            samples = samples.reshape(B, H, W, 3).permute(0, 3, 1, 2)
            return samples

    def sample_n(self, features, n_samples=50):
        """
        Draw n_samples Monte Carlo samples from p(E,B | x) in a single batch.

        features:  [B, C, H, W]  – encoder features
        n_samples: int           – number of samples to draw per pixel

        Returns:
            samples: [n_samples, B, 2, H, W]
        """
        B, C, H, W = features.shape
        yy, xx = torch.meshgrid(
                torch.linspace(-1,1,H,device=features.device),
                torch.linspace(-1,1,W,device=features.device), indexing="ij"
        )
        coords = torch.stack([xx, yy]).expand(B, -1, H, W)
        features = torch.cat([features, coords], dim=1)
        context = self.context_net(features).permute(0, 2, 3, 1).reshape(-1, self.context_dim)

        n_pix = B * H * W
        z = self.base.sample(n_samples * n_pix)
        z = z.view(n_samples * n_pix, -1)
        context_rep = context.repeat(n_samples, 1)

        # Invert the flow transform
        samples, _ = self.flow._transform.inverse(z, context=context_rep)

        # Reshape to [n_samples, B, 2, H, W]
        samples = samples.view(n_samples, B, H, W, 3).permute(0, 1, 4, 2, 3)
        return samples

        
class ResidualBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, pool_type="avg", dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(p=dropout_p) 

        # Skip connection if channel mismatch
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1)

        # --- Pooling variants ---
        self.pool_type = pool_type
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool_type == "avgmax":
            # Concatenate avg + max → doubles channels for fc1
            self.pool_avg = nn.AdaptiveAvgPool2d(1)
            self.pool_max = nn.AdaptiveMaxPool2d(1)
        elif pool_type == "learned":
            # 1x1 conv to learn pooling weights (H×W → 1)
            self.pool = nn.Conv2d(out_channels, 1, kernel_size=1)

        # --- SE MLP ---
        if pool_type == "avgmax":
            se_in = 2 * out_channels
        else:
            se_in = out_channels

        self.fc1 = nn.Linear(se_in, out_channels // reduction)
        self.fc2 = nn.Linear(out_channels // reduction, out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # --- SE attention pooling ---
        if self.pool_type == "avg":
            w = self.pool(out).view(out.size(0), -1)
        elif self.pool_type == "max":
            w = self.pool(out).view(out.size(0), -1)
        elif self.pool_type == "avgmax":
            w_avg = self.pool_avg(out).view(out.size(0), -1)
            w_max = self.pool_max(out).view(out.size(0), -1)
            w = torch.cat([w_avg, w_max], dim=1)
        elif self.pool_type == "learned":
            # Apply learned 1x1 conv → softmax over spatial dims
            weights = F.softmax(self.pool(out).view(out.size(0), -1), dim=1)
            w = torch.sum(out.view(out.size(0), out.size(1), -1) * weights.unsqueeze(1), dim=-1)

        # --- SE excitation ---
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).view(out.size(0), out.size(1), 1, 1)
        out = out * w

        # Skip connection
        if self.proj is not None:
            identity = self.proj(identity)
        out += identity
        return F.relu(out)

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


class main_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_channels=32,
                 use_fc_bottleneck=True, fc_hidden=512, fc_spatial=4, rotation_prob=0,
                 use_cross_attention=False, attn_heads=1, epochs=epochs, pool_type='max', 
                 err_L1=1, err_Multi=1,err_Pear=1,err_SSIM=1,err_Grad=1,err_Power=1,err_Cross=0,
                 suffix='', dropout_1=0, dropout_2=0, dropout_3=0, use_one_hot=True,
                rng=[0,1], num_bins=32):
        super().__init__()
        arg_dict = locals()
        #self.range=rng
        #self.num_bins=num_bins
        self.use_fc_bottleneck = use_fc_bottleneck
        self.fc_spatial = fc_spatial
        self.dropout_2=dropout_2
        self.use_cross_attention=use_cross_attention
        self.err_L1=err_L1
        self.err_Multi=err_Multi
        self.err_Pear=err_Pear
        self.err_SSIM=err_SSIM
        self.err_Grad=err_Grad
        self.err_Power=err_Power
        self.err_Cross=err_Cross
        self.rotation_prob=rotation_prob
        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))
        self.register_buffer('num_bins',torch.tensor(num_bins, dtype=torch.int))
        self.register_buffer('range',torch.tensor(rng))
        if 0:
            for arg in arg_dict:
                if arg in ['self','__class__','arg_dict','text','data']:
                    continue
                if type(arg_dict[arg]) == str:
                    text = arg_dict[arg]
                    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)
                else:
                    data = torch.tensor(arg_dict[arg])
                self.register_buffer(arg,data)

        #raise
        #self.use_fc_bottleneck = use_fc_bottleneck
        #self.use_cross_attention = use_cross_attention

        # Encoder
        self.enc1 = ResidualBlockSE(in_channels, base_channels, pool_type=pool_type, dropout_p=dropout_1)
        self.enc2 = ResidualBlockSE(base_channels, base_channels*2, pool_type=pool_type, dropout_p=dropout_1)
        self.enc3 = ResidualBlockSE(base_channels*2, base_channels*4, pool_type=pool_type, dropout_p=dropout_1)
        self.enc4 = ResidualBlockSE(base_channels*4, base_channels*8, pool_type=pool_type, dropout_p=dropout_1)
        self.pool = nn.MaxPool2d(2)

        # Optional FC bottleneck
        if use_fc_bottleneck:
            self.fc1 = nn.Linear(base_channels*8*fc_spatial*fc_spatial, fc_hidden)
            self.fc2 = nn.Linear(fc_hidden, base_channels*8*fc_spatial*fc_spatial)

        # Learned upsampling via ConvTranspose2d
        self.up4 = nn.ConvTranspose2d(base_channels*8, base_channels*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Decoder with skip connections
        self.dec4 = ResidualBlockSE(base_channels*8 + base_channels*4, base_channels*4, pool_type=pool_type, dropout_p=dropout_3)
        self.dec3 = ResidualBlockSE(base_channels*4 + base_channels*2, base_channels*2, pool_type=pool_type, dropout_p=dropout_3)
        self.dec2 = ResidualBlockSE(base_channels*2 + base_channels, base_channels, pool_type=pool_type, dropout_p=dropout_3)
        self.dec1 = nn.Conv2d(base_channels, out_channels, 3, padding=1)

        # --- Probability distribution output head
        self.flow_head = EBFlowHead(base_channels, hidden_features=128, num_layers=8)
        #self.flow_head = EBFlowHead(in_channels=224, hidden_features=256, num_layers=8)


        # Optional cross-attention
        if use_cross_attention:
            self.cross_attn = CrossAttention(out_channels, num_heads=attn_heads)



    def forward(self, x, return_features=False):
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
            z = F.dropout(z, p=self.dropout_2, training=self.training)
            z = F.relu(self.fc2(z))
            z = F.dropout(z, p=self.dropout_2, training=self.training)
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


        context_in = d2
        if 0:
            context_in = torch.cat([
                    d2,
                    F.interpolate(d3, size=d2.shape[-2:], mode="bilinear", align_corners=False),
                    F.interpolate(d4, size=d2.shape[-2:], mode="bilinear", align_corners=False)
            ], dim=1)
        if self.use_cross_attention:
            out_main = self.cross_attn(out_main)

        if self.training or return_features:
            # Expect target passed separately in criterion
            return out_main, context_in  # pass decoder features to criterion
        else:
            # Sample predicted E,B map from flow
            samples = self.flow_head(context_in, mode="sample")
            return out_main, samples
    def criterion1(self, preds, target, epoch=10):
        """
        preds: tuple (out_main, features)
        target: [B, C, H, W]
        """
        out_main, features = preds
        all_loss = {}

        # 1. Flow negative log-likelihood loss on E,B
        flow_loss = self.flow_head(features.detach(), target=target, mode="train")
        err_Flow = max([epoch/30,1])*0.1
        all_loss['Flow'] = err_Flow*flow_loss

        # 2. Optionally keep your auxiliary image-space losses
        if self.err_L1 > 0:
            all_loss['L1'] = self.err_L1 * F.l1_loss(out_main, target)
        if self.err_SSIM > 0:
            all_loss['SSIM'] = self.err_SSIM * ssim_loss(out_main[:,1:,:,:], target[:,1:,:,:])
        if self.err_Grad > 0:
            all_loss['Grad'] = self.err_Grad * gradient_loss(out_main[:,1:,:,:], target[:,1:,:,:])
        if self.err_Pear > 0:
            all_loss['Pear'] = self.err_Pear * pearson_loss(out_main[:,1:,:,:], target[:,1:,:,:])
        if self.err_Power > 0:
            all_loss['Power'] = self.err_Power * power_spectra_crit(out_main, target)
        if self.err_Cross > 0:
            all_loss['Cross'] = self.err_Cross * cross_spectra_crit(out_main, target)

        return all_loss



        return all_loss
    def criterion(self, preds, target, epoch=10):
        losses = self.criterion1(preds,target,epoch=epoch)

        return sum(losses.values())
