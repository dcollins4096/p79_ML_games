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



idd = 130
what = "cross attention"

fname_train = "p79d_subsets_S512_N5_xyz_down_128_2356_x.h5"
fname_valid = "p79d_subsets_S512_N5_xyz_down_128_4_x.h5"
#ntrain = 2000
ntrain = 1000
#ntrain = 600
#nvalid=3
ntrain = 10
nvalid=30
downsample = 64
#device = device or ("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs  = 200
epochs = 20
lr = 1e-3
#lr = 1e-4
batch_size=1
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
    scaler = torch.amp.GradScaler('cuda')

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
            with torch.amp.autocast('cuda'):
                preds = model(xb)
                if verbose:
                    print("  crit")

                #loss  = model.criterion(preds, yb[:,0:1,:,:])
                loss  = model.criterion(preds, yb)

            if verbose:
                print("  scale backward")
            #loss.backward()
            scaler.scale(loss).backward()
            

            if verbose:
                print("  steps")
            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()

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

class CrossAttentionPooledKV(nn.Module):
    def __init__(self, channels, num_heads=4, kv_pool_factor=4):
        """
        kv_pool_factor: how much spatially to downsample K and V (integer).
                       kv_pool_factor=4 -> (H*W) -> (H/4 * W/4)
        """
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.channels = channels
        self.kv_pool_factor = kv_pool_factor

        # We'll use linear projections for Q,K,V and then a standard scaled dot-product
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        q = self.q_proj(x)                # [B, C, H, W]

        # Pool K and V spatially to reduce sequence length
        if self.kv_pool_factor > 1:
            k_src = F.adaptive_avg_pool2d(x, (H // self.kv_pool_factor, W // self.kv_pool_factor))
            v_src = k_src
        else:
            k_src = x
            v_src = x

        k = self.k_proj(k_src)            # [B, C, Hk, Wk]
        v = self.v_proj(v_src)            # [B, C, Hk, Wk]

        # reshape to [B, heads, seq, head_dim]
        head_dim = C // self.num_heads
        def reshape_for_attn(t, Ht, Wt):
            # t: [B, C, Ht, Wt] -> [B, heads, Ht*Wt, head_dim]
            t = t.view(B, self.num_heads, head_dim, Ht * Wt).permute(0,1,3,2)
            return t

        q_flat = reshape_for_attn(q, H, W)             # [B, h, S_q, d]
        k_flat = reshape_for_attn(k, k.shape[-2], k.shape[-1])  # [B, h, S_k, d]
        v_flat = reshape_for_attn(v, k.shape[-2], k.shape[-1])  # [B, h, S_k, d]

        # scaled dot product: [B, h, S_q, S_k]
        scale = 1.0 / math.sqrt(head_dim)
        attn_logits = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * scale
        attn = torch.softmax(attn_logits, dim=-1)

        # output: [B, h, S_q, d]
        out = torch.matmul(attn, v_flat)

        # merge heads back: [B, C, H, W]
        out = out.permute(0,1,3,2).contiguous().view(B, C, H, W)
        out = self.out_proj(out)
        return out


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
import torch.nn as nn
import torch.nn.functional as F

# --- Cross-Attention Block ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, q_channels, kv_channels, num_heads=4, reduction=2):
        super().__init__()
        d_model = q_channels // reduction
        self.q_proj = nn.Conv2d(q_channels, d_model, kernel_size=1)
        self.k_proj = nn.Conv2d(kv_channels, d_model, kernel_size=1)
        self.v_proj = nn.Conv2d(kv_channels, d_model, kernel_size=1)

        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.out_proj = nn.Conv2d(d_model, q_channels, kernel_size=1)

    def forward(self, q_feat, kv_feat):
        B, Cq, H, W = q_feat.shape
        _, Ckv, Hk, Wk = kv_feat.shape

        # Match spatial resolution if needed
        if (Hk, Wk) != (H, W):
            kv_feat = F.interpolate(kv_feat, size=(H, W), mode="bilinear", align_corners=False)

        Q = self.q_proj(q_feat)   # [B, d_model, H, W]
        K = self.k_proj(kv_feat)  # [B, d_model, H, W]
        V = self.v_proj(kv_feat)

        # Flatten
        Q = Q.flatten(2).transpose(1, 2)  # [B, HW, d_model]
        K = K.flatten(2).transpose(1, 2)
        V = V.flatten(2).transpose(1, 2)

        attn_out, _ = self.attn(Q, K, V)

        attn_out = attn_out.transpose(1, 2).view(B, -1, H, W)
        return q_feat + self.out_proj(attn_out)


# --- Your Main Net with Multi-Scale Supervision and Cross-Attention ---
class main_net(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, out_channels=1, use_cross_attention=True):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        # Example encoder
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.enc3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1)

        # Example decoder
        self.dec3 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1)
        self.dec2 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)

        # Cross-attention fusion block
        if use_cross_attention:
            self.cross_attn = CrossAttentionBlock(
                q_channels=base_channels,   # decoder output channels
                kv_channels=base_channels * 4,  # encoder skip (e3) channels
                num_heads=2,
                reduction=2
            )

        # Output heads for multi-scale supervision
        self.out_head_main = nn.Conv2d(base_channels, out_channels, 1)
        self.out_head_aux2 = nn.Conv2d(base_channels * 2, out_channels, 1)
        self.out_head_aux3 = nn.Conv2d(base_channels * 4, out_channels, 1)

    def forward(self, x):
        # --- Encoder ---
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(F.max_pool2d(e1, 2)))
        e3 = F.relu(self.enc3(F.max_pool2d(e2, 2)))

        # --- Decoder ---
        d3 = F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = F.relu(self.dec3(d3))
        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = F.relu(self.dec2(d2))

        # --- Cross-Attention Fusion ---
        if self.use_cross_attention:
            d2 = self.cross_attn(d2, e3)

        # --- Multi-Scale Outputs ---
        out_main = self.out_head_main(d2)
        out_aux2 = self.out_head_aux2(d3)
        out_aux3 = self.out_head_aux3(e3)

        return out_main, out_aux2, out_aux3

    def criterion1(self, preds, target):
        """
        preds: tuple of (out_main, out_d2, out_d3, out_d4)
        target: [B, C, H, W] ground truth
        """
        out_main, out_d2, out_d3  = preds

        # Downsample target to match each prediction
        t_d2 = F.interpolate(target, size=out_d2.shape[-2:], mode="bilinear", align_corners=False)
        t_d3 = F.interpolate(target, size=out_d3.shape[-2:], mode="bilinear", align_corners=False)
        #t_d4 = F.interpolate(target, size=out_d4.shape[-2:], mode="bilinear", align_corners=False)

        loss_main = F.l1_loss(out_main, target)
        loss_d2   = F.l1_loss(out_d2, t_d2)
        loss_d3   = F.l1_loss(out_d3, t_d3)
        #loss_d4   = F.l1_loss(out_d4, t_d4)

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


        stuff=[loss_main, loss_d2, loss_d3, lambda_ssim, lambda_grad, lambda_pear]
        out = torch.stack(stuff).to(device)
        return out

    def criterion(self, preds, target):
        losses = self.criterion1(preds,target)
        weights = torch.tensor([1,0.5,0.25,1,1,2]).to(device)
        return (losses*weights).sum()

