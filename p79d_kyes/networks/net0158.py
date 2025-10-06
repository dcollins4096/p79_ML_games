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


idd = 158
what = "155 with rotational equivariant network"

fname_train = "p79d_subsets_S256_N10_xyz_down_128_rot23456_second.h5"
fname_valid = "p79d_subsets_S256_N10_xyz_down_128_rot23456_first.h5"
#ntrain = 2000
#ntrain = 1000 #ntrain = 600
#ntrain = 20
ntrain = 5000
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
print("NTRAIN",ntrain)
def load_data():

    print('read the data')
    train= loader.loader(fname_train,ntrain=ntrain, nvalid=nvalid)
    valid= loader.loader(fname_valid,ntrain=5500, nvalid=nvalid)
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
    print('START TRAINER')
    set_seed()

    #ds_train = SphericalDataset(all_data['train'], rotation_prob=model.rotation_prob)
    #ds_val   = SphericalDataset(all_data['valid'], rotation_prob=model.rotation_prob)
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
    save_err_Bisp = -1
    #if model.err_Bisp > 0:
    #    save_err_Bisp = model.err_Bisp
    #    model.err_Bisp = torch.tensor(0.0)


    for epoch in range(1, epochs+1):
        if epoch > 50 and save_err_Bisp>0:
            model.err_Bisp = save_err_Bisp
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
    err_B=0
    if len(guess) == 3:
        err_B = power_spectrum_delta(guess[:,2:3,:,:], target[:,2:3,:,:])
    return err_T+err_E+err_B

import bispectrum
def bispectrum_crit(guess,target):
    nsamples=100
    T_guess = bispectrum.compute_bispectrum_torch(guess[:,0:1,:,:]  ,nsamples=nsamples)[0]
    T_target = bispectrum.compute_bispectrum_torch(target[:,0:1,:,:],nsamples=nsamples)[0]
    dT = torch.mean(torch.abs(torch.log(torch.abs( T_guess / T_target))))

    E_guess = bispectrum.compute_bispectrum_torch(guess[:,1:2,:,:]  ,nsamples=nsamples)[0]
    E_target = bispectrum.compute_bispectrum_torch(target[:,1:2,:,:],nsamples=nsamples)[0]
    dE = torch.mean(torch.abs(torch.log(torch.abs( E_guess / E_target))))
    dB = 0
    if len(guess)==3:
        B_guess = bispectrum.compute_bispectrum_torch(guess[:,2:3,:,:]  ,nsamples=nsamples)[0]
        B_target = bispectrum.compute_bispectrum_torch(target[:,2:3,:,:],nsamples=nsamples)[0]
        dB = torch.mean(torch.abs(torch.log(torch.abs( B_guess / B_target))))
    #pdb.set_trace()
    return dT+dE+dB

def error_real_imag(guess,target):

    L1  = F.l1_loss(guess.real, target.real)
    L1 += F.l1_loss(guess.imag, target.imag)
    return L1

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn

# -------------------------------------------------
#  Residual Squeeze–Excitation Block (E(2)-equivariant)
# -------------------------------------------------
class ResidualBlockSE_E2(nn.Module):
    def __init__(self, r2_act, in_type, out_channels, kernel_size=3, reduction=16, dropout_p=0.0):
        super().__init__()
        self.r2_act = r2_act
        self.in_type = in_type
        self.out_type = enn.FieldType(r2_act, out_channels * [r2_act.regular_repr])

        self.block = enn.SequentialModule(
            enn.R2Conv(in_type, self.out_type, kernel_size=kernel_size, padding=1, bias=False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type, inplace=True),
            enn.R2Conv(self.out_type, self.out_type, kernel_size=kernel_size, padding=1, bias=False),
            enn.InnerBatchNorm(self.out_type),
        )

        # 1x1 skip conv if shape changes
        if in_type != self.out_type:
            self.skip = enn.R2Conv(in_type, self.out_type, kernel_size=1, bias=False)
        else:
            self.skip = None

        # SE
        self.out_type = self.out_type  # or whatever your last conv’s out_type is
        c = self.out_type.size          # number of feature channels (representation components)

        self.fc1 = nn.Conv2d(c, c // 4, 1, bias=True)
        self.fc2 = nn.Conv2d(c // 4, c, 1, bias=True)
        #self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, 1)
        #self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, 1)
        self.relu = enn.ReLU(self.out_type, inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.skip is not None:
            identity = self.skip(identity)
        out = out + identity
        out = self.relu(out)

        # Squeeze–Excitation (on tensor level)
        t = out.tensor
        w = t.mean(dim=(2, 3), keepdim=True)
        w = torch.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        t = t * w
        out = enn.GeometricTensor(t, out.type)

        out.tensor = self.dropout(out.tensor)
        return out

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





# -------------------------------------------------
#  Full U-Net with skip connections and spin-2 output
# -------------------------------------------------
#from e2cnn.group import cached_basisexpansions
#cached_basisexpansions.set_cache_dir("./e2cnn_basis_cache")
#cached_basisexpansions.set_cache_mode("disk")
class main_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_channels=16,
                 use_fc_bottleneck=True, fc_hidden=512, fc_spatial=4, rotation_prob=0,
                 use_cross_attention=False, attn_heads=1, epochs=epochs, pool_type='max', 
                 err_L1=1, err_Multi=0,err_Pear=1,err_SSIM=1,err_Grad=1,err_Power=1,err_Bisp=1,
                 suffix='', dropout_1=0, dropout_2=0, dropout_3=0, N=4):
        super().__init__()
        print('dork')
        self.r2_act = gspaces.Rot2dOnR2(N=N)
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
        self.err_Bisp=err_Bisp

        # Spin-0 input (scalar)
        print('dork1')
        in_type = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])

        # Encoder
        self.enc1 = ResidualBlockSE_E2(self.r2_act, in_type, base_channels, dropout_p=dropout_1)
        self.pool1 = enn.PointwiseAvgPoolAntialiased(self.enc1.out_type, sigma=0.66, stride=2)
        print('dork2')

        self.enc2 = ResidualBlockSE_E2(self.r2_act, self.enc1.out_type, base_channels * 2, dropout_p=dropout_1)
        print('dork2.1')
        self.pool2 = enn.PointwiseAvgPoolAntialiased(self.enc2.out_type, sigma=0.66, stride=2)
        print('dork2.2')

        self.enc3 = ResidualBlockSE_E2(self.r2_act, self.enc2.out_type, base_channels * 4, dropout_p=dropout_1)
        print('dork2.3')
        self.pool3 = enn.PointwiseAvgPoolAntialiased(self.enc3.out_type, sigma=0.66, stride=2)
        print('dork3')

        self.enc4 = ResidualBlockSE_E2(self.r2_act, self.enc3.out_type, base_channels * 8, dropout_p=dropout_1)
        print('dork3.1')
        self.pool4 = enn.PointwiseAvgPoolAntialiased(self.enc4.out_type, sigma=0.66, stride=2)
        print('dork3.2')

        # Bottleneck
        self.bottleneck = ResidualBlockSE_E2(self.r2_act, self.enc4.out_type, base_channels * 8, dropout_p=dropout_2)
        print('dork4')

        # Decoder (with skip connections)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = ResidualBlockSE_E2(self.r2_act, self.bottleneck.out_type, base_channels * 8, dropout_p=dropout_3)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = ResidualBlockSE_E2(self.r2_act, self.dec4.out_type, base_channels * 4, dropout_p=dropout_3)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ResidualBlockSE_E2(self.r2_act, self.dec3.out_type, base_channels * 2, dropout_p=dropout_3)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ResidualBlockSE_E2(self.r2_act, self.dec2.out_type, base_channels, dropout_p=dropout_3)
        print('dork5')

        # Final spin-2 output
        spin2_repr = self.r2_act.irrep(2)
        out_type = enn.FieldType(self.r2_act, out_channels * [spin2_repr])
        self.out_conv = enn.R2Conv(self.dec1.out_type, out_type, kernel_size=3, padding=1)
        print('finished')
        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = enn.GeometricTensor(x, self.enc1.in_type)

        # --- Encoder ---
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # --- Decoder with skip connections ---
        d4 = self.dec4(enn.GeometricTensor(self.up4(b.tensor), self.dec4.in_type))
        d4.tensor = d4.tensor + e4.tensor  # skip connection (sum)
        d3 = self.dec3(enn.GeometricTensor(self.up3(d4.tensor), self.dec3.in_type))
        d3.tensor = d3.tensor + e3.tensor
        d2 = self.dec2(enn.GeometricTensor(self.up2(d3.tensor), self.dec2.in_type))
        d2.tensor = d2.tensor + e2.tensor
        d1 = self.dec1(enn.GeometricTensor(self.up1(d2.tensor), self.dec1.in_type))
        d1.tensor = d1.tensor + e1.tensor

        out = self.out_conv(d1)
        return out.tensor,

    def criterion1(self, preds, target1):
        """
        preds: tuple of (out_main, out_d2, out_d3, out_d4)
        target: [B, C, H, W] ground truth
        """
        all_loss = {}

        if len(preds) == 4:
            out_main, out_d2, out_d3, out_d4 = preds
        else:
            out_main = preds[0]
        if out_main.shape[1] == 3:
            do_t=True
        elif out_main.shape[1] == 2:
            do_t=False
        else:
            print('problem')
            pdb.set_trace()
        if do_t:
            target=target1
        else:
            target = target1[:,1:,:,:]
        # Downsample target to match each prediction
        if self.err_L1>0:
            loss_main = F.l1_loss(out_main, target)
            all_loss['L1_0']=self.err_L1*loss_main
        if self.err_Multi>0:
            t_d2 = F.interpolate(target, size=out_d2.shape[-2:], mode="bilinear", align_corners=False)
            t_d3 = F.interpolate(target, size=out_d3.shape[-2:], mode="bilinear", align_corners=False)
            t_d4 = F.interpolate(target, size=out_d4.shape[-2:], mode="bilinear", align_corners=False)

            loss_d2   = F.l1_loss(out_d2, t_d2)
            loss_d3   = F.l1_loss(out_d3, t_d3)
            loss_d4   = F.l1_loss(out_d4, t_d4)
            loss_multi = self.err_Multi*(loss_d2+loss_d3+loss_d4)
            all_loss['L1_Multi'] = loss_multi

        # Weighted sum (more weight on full-res output)
        if self.err_SSIM > 0:
            ssim_t  = ssim_loss(out_main[:,0:1,:,:], target[:,0:1,:,:])
            ssim_e  = ssim_loss(out_main[:,1:2,:,:], target[:,1:2,:,:])
            ssim_b=0
            if do_t:
                ssim_b  = ssim_loss(out_main[:,2:3,:,:], target[:,2:3,:,:])

            lambda_ssim = self.err_SSIM*(ssim_e+ssim_b+ssim_t)/3
            all_loss['SSIM']=lambda_ssim
        if self.err_Grad > 0:
            grad_t  = gradient_loss(out_main[:,0:1,:,:], target[:,0:1,:,:])
            grad_e  = gradient_loss(out_main[:,1:2,:,:], target[:,1:2,:,:])
            grad_b=0
            if do_t:
                grad_b  = gradient_loss(out_main[:,2:3,:,:], target[:,2:3,:,:])
            lambda_grad = self.err_Grad*(grad_e+grad_b+grad_t)/3
            all_loss['Grad']=lambda_grad
        if self.err_Pear > 0:
            pear_t  = pearson_loss(out_main[:,0:1,:,:], target[:,0:1,:,:])
            pear_e  = pearson_loss(out_main[:,1:2,:,:], target[:,1:2,:,:])
            pear_b=0
            if do_t:
                pear_b  = pearson_loss(out_main[:,2:3,:,:], target[:,2:3,:,:])
            lambda_pear = self.err_Pear*(pear_e+pear_b+pear_t)/3
            all_loss['Pear']=lambda_pear
        if self.err_Power > 0:
            lambda_power = self.err_Power*power_spectra_crit(out_main, target)
            all_loss['Power'] = lambda_power
        if self.err_Bisp > 0:
            lambda_bisp = self.err_Bisp*bispectrum_crit(out_main,target)
            all_loss['Bisp'] = lambda_bisp

        return all_loss

    def criterion(self, preds, target):
        losses = self.criterion1(preds,target)

        return sum(losses.values())

