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


idd = 4008
what = "4007 with cos annealing and some other tweaks."

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
ntrain = 14000
#nvalid=3
#ntrain = 10
nvalid=30
ntest = 5000
downsample = 64
#device = device or ("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
#epochs  = 1e6
epochs = 30
lr = 0.5e-3
#lr = 1e-4
batch_size=16
lr_schedule=[1000]
weight_decay = 5e-2
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

    model = main_net(img_size=downsample,base_channels=16,fc_hidden=512 , fc_spatial=4, use_fc_bottleneck=fc_bottleneck, out_channels=3, use_cross_attention=False, attn_heads=1)#, dropout_1=0.3, dropout_2=0.3, dropout_3=0.3)

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
    def __init__(self, all_data, quan, rand=False):
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
        k = torch.randint(0, 4, (1,)).item()
        theset = torch.rot90(theset, k, dims=[-2, -1])
        #theset[0] = torch.log(theset[0])
        ms = self.quan['Ms_act'][idx]
        ma = self.quan['Ma_act'][idx]
        target = torch.log(torch.tensor([ms], dtype=torch.float32).to(device))
        return theset[0:3].to(device), target

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
    if 0:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_schedule, #[100,300,600],  # change after N and N+M steps
            gamma=0.1             # multiply by gamma each time
        )
    if 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs, 
            eta_min=1e-7
        )

    best_val = float("inf")
    best_state = None
    load_best = False
    patience = 1e6
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
        print(f"Load Best. Best val {best_val:.4f}.")
        model.load_state_dict(best_state)
    return model

    # quick plot (optional)


class ResidualBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, pool_type="avg", dropout_p=0.0, dilation=1):
        super().__init__()
        padding=dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        #self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=padding, dilation=dilation)
        #self.bn2 = nn.BatchNorm2d(out_channels)

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
        #out = self.bn2(self.conv2(out))

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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------
# 1. Helper: Differentiable Radial Profile (PSD)
# ---------------------------------------------------------
class RadialProfile(nn.Module):
    def __init__(self, size, n_bins=None):
        super().__init__()
        H, W = size, size 
        y, x = np.indices((H, W))
        center = (H // 2, W // 2)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # We need the maximum radius (corner) to size the buffer correctly
        r = r.astype(int)
        max_r = r.max()
        
        # User defined bins (usually up to the edge, not corner)
        if n_bins is None:
            n_bins = int(min(H, W) / 2)
        
        self.n_bins = n_bins
        self.r_flat = torch.from_numpy(r.flatten()).long()
        
        # Calculate bin counts up to the corner (max_r)
        bin_count = torch.bincount(self.r_flat, minlength=max_r + 1)
        
        # Avoid division by zero
        bin_count[bin_count == 0] = 1 
        
        self.register_buffer('bin_count', bin_count.float())
        self.register_buffer('indices', self.r_flat)
        # Save the size needed for the temporary output buffer
        self.max_r = max_r

    def forward(self, x):
        # x shape: [B, 1, H, W] or [B, H, W]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        B, C, H, W = x.shape
        
        # 1. FFT
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft)
        power_spectrum = torch.abs(fft)**2
        
        # Collapse channels if C > 1 (safety for the future)
        if C > 1:
            power_spectrum = power_spectrum.mean(dim=1)
        else:
            power_spectrum = power_spectrum.squeeze(1)
            
        # Flatten: [B, H*W]
        ps_flat = power_spectrum.view(B, -1)
        
        # 2. Accumulate
        # Create output buffer large enough for corners (max_r + 1)
        output = torch.zeros(B, self.max_r + 1, device=x.device)
        
        for b in range(B):
            output[b].index_add_(0, self.indices, ps_flat[b])
            
        # Normalize
        output = output / self.bin_count
        
        # 3. Return
        # We only return bins 1 to n_bins (ignoring DC component 0, and ignoring corners)
        # Add epsilon to avoid log(0)
        return torch.log(output[:, 1:self.n_bins] + 1e-8)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FFTCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=128):
        super().__init__()
        # Total input channels = 3 (mag) + 3 (sin) + 3 (cos) = 9
        self.input_channels = in_channels * 3
        
        self.conv_net = nn.Sequential(
            # Start with BatchNorm to balance mag vs sin/cos scales
            nn.BatchNorm2d(self.input_channels),
            
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2), # 64 -> 32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2), # 32 -> 16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, 3, 64, 64]
        
        # 1. Compute FFT and Shift
        ffted = torch.fft.fft2(x, dim=(-2, -1))
        ffted = torch.fft.fftshift(ffted, dim=(-2, -1))
        
        # 2. Extract Magnitude and Phase
        mag = torch.abs(ffted)
        phase = torch.angle(ffted) # Returns values in [-pi, pi]

        B, C, H, W = mag.shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        dist = torch.sqrt(x**2 + y**2).to(mag.device)
        mask = (dist > 0.5).float() # Keep 80% of the frequency radius
        mask = 1.0

        mag = mag * mask
        phase = phase * mask
        
        # 3. Process Magnitude
        # Zero out DC to focus on fluctuations
        mag[:, :, mag.shape[-2]//2, mag.shape[-1]//2] = 0
        mag_log = torch.log1p(mag)
        #mag_log = mag
        
        # 4. Process Phase with Sin/Cos encoding
        phase_sin = torch.sin(phase)
        phase_cos = torch.cos(phase)
        
        # 5. Concatenate to 9 channels: [B, 9, 64, 64]
        # Order: [Mag_0, Mag_1, Mag_2, Sin_0, Sin_1, Sin_2, Cos_0, Cos_1, Cos_2]
        fft_input = torch.cat([mag_log, phase_sin, phase_cos], dim=1)
        
        # 6. CNN pass
        z = self.conv_net(fft_input)
        z = z.view(z.size(0), -1)
        return self.fc(z)
# ---------------------------------------------------------
# 2. Updated Main Net
class main_net(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32,
                 use_fc_bottleneck=True, fc_hidden=2048, fc_spatial=4, rotation_prob=0,
                 use_cross_attention=False, attn_heads=1, epochs=100, pool_type='max', 
                 suffix='', dropout_1=0.2, dropout_2=0.2, dropout_3=0.2, 
                 predict_scalars=True, n_scalars=1, img_size=64):
        super().__init__()
        
        self.predict_log=True
        self.predict_scalars=True
        self.n_scalars=1
        self.use_fc_bottleneck = use_fc_bottleneck
        self.fc_spatial = fc_spatial
        
        # --- Spatial Encoder (CNN) ---
        #d1, d2, d3, d4 = 2, 4, 8, 16
        d1,d2,d3,d4=1,1,1,1
        self.enc1 = ResidualBlockSE(in_channels, base_channels, pool_type=pool_type, dropout_p=dropout_1, dilation=d1)
        self.enc2 = ResidualBlockSE(base_channels, base_channels*2, pool_type=pool_type, dropout_p=dropout_1, dilation=d2)
        self.enc3 = ResidualBlockSE(base_channels*2, base_channels*4, pool_type=pool_type, dropout_p=dropout_1, dilation=d3)
        self.enc4 = ResidualBlockSE(base_channels*4, base_channels*8, pool_type=pool_type, dropout_p=dropout_1, dilation=d4)
        self.pool = nn.MaxPool2d(2)

        # Calculate CNN output size
        # base=32 -> enc4=256 channels.
        # adaptive_pool to (fc_spatial, fc_spatial) -> 256 * 4 * 4
        cnn_raw_dim = base_channels * 2 * fc_spatial * fc_spatial 
        
        # FIX 1: CNN Bottleneck
        # Compress huge spatial vector down to 128 BEFORE fusion
        Nmid = 16
        self.cnn_bottleneck = nn.Sequential(
            nn.Linear(cnn_raw_dim, Nmid),
            nn.LayerNorm(Nmid),  # LayerNorm is often better for regression stability
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.fft_encoder = FFTCNNEncoder(in_channels=3, out_dim=Nmid)



        # --- Fusion & Regression ---
        # Now we fuse 128 (CNN) + 128 (PSD) = 256
        total_in_dim = 2*Nmid

        self.regressor = nn.Sequential(
            nn.Linear(total_in_dim, fc_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout_2),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.GELU(),
            nn.Linear(fc_hidden // 2, n_scalars)
        )
        
        # Learnable gate (initialized to small value to start with primarily CNN)
        self.psd_weight = nn.Parameter(torch.tensor(1.0))

        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))

    def forward(self, x, cache=False):
            # x shape: [B, 3, 64, 64]
            # Ch 0: Density, Ch 1: Velocity Centroid, Ch 2: Velocity Variance
            
            # 1. Physical Pre-processing (Anti-NaN)
            x_proc = x.clone()
            x_proc[:, 0] = torch.log(x[:, 0]) # Density
            # Symmetric Log for Velocity Centroid
            x_proc[:, 1] = torch.sign(x[:, 1]) * torch.log1p(torch.abs(x[:, 1])) 
            x_proc[:, 2] = torch.log(x[:, 2]) # Variance

            # 2. Path A: Spatial CNN (Use processed data)
            e1 = self.enc1(x_proc)
            e2 = self.enc2(self.pool(e1))
            #e3 = self.enc3(self.pool(e2))
            #e4 = self.enc4(self.pool(e3))
            
            z_spatial = F.adaptive_avg_pool2d(e2, (self.fc_spatial, self.fc_spatial))
            z_spatial = z_spatial.reshape(z_spatial.size(0), -1)
            z_spatial = self.cnn_bottleneck(z_spatial)

            # 3. Path B: FFT CNN (Use processed data)
            z_freq = self.fft_encoder(x_proc)
            #z_freq = self.fft_encoder(x)

            # 4. Path C: Fusion
            z_combined = torch.cat([z_spatial, z_freq], dim=1)
            if cache:
                self.z_freq = z_freq
                self.z_spatial = z_spatial
            
            return self.regressor(z_combined)

    def criterion1(self, preds, target):
        target = torch.clamp(target, min=1e-6)
        # Assuming model predicts log(M), and target is raw M
        return {'mse': F.mse_loss(preds, target)}

    def criterion(self, preds, target):
        losses = self.criterion1(preds, target)
        return sum(losses.values())
