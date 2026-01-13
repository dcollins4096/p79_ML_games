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


idd = 4003
what = "4000 with old stuff cut out to not confuse GPT"

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
downsample = 64
#device = device or ("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
#epochs  = 1e6
epochs = 50
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

    model = main_net(base_channels=32,fc_hidden=2048 , fc_spatial=8, use_fc_bottleneck=fc_bottleneck, out_channels=3, use_cross_attention=False, attn_heads=1)#, dropout_1=0.3, dropout_2=0.3, dropout_3=0.3)

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

    ds_train = SphericalDataset(all_data['train'],all_data['quantities']['train'], rotation_prob=model.rotation_prob, rand=True)
    ds_val   = SphericalDataset(all_data['valid'],all_data['quantities']['valid'], rotation_prob=model.rotation_prob)
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


class main_net(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32,
                 use_fc_bottleneck=True, fc_hidden=512, fc_spatial=4, rotation_prob=0,
                 use_cross_attention=True, attn_heads=1, epochs=epochs, pool_type='max', 
                 suffix='', dropout_1=0, dropout_2=0, dropout_3=0, predict_scalars=True, n_scalars=1):
        super().__init__()
        self.use_fc_bottleneck = use_fc_bottleneck
        self.fc_spatial = fc_spatial
        self.dropout_2=dropout_2
        self.use_cross_attention=use_cross_attention
        self.rotation_prob=rotation_prob
        self.predict_scalars = predict_scalars
        self.predict_scalars_only = True
        self.n_scalars = n_scalars
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

        d1, d2, d3, d4 = 2, 4, 8,16
        # Encoder
        self.enc1 = ResidualBlockSE(in_channels, base_channels, pool_type=pool_type, dropout_p=dropout_1, dilation=d1)
        self.enc2 = ResidualBlockSE(base_channels, base_channels*2, pool_type=pool_type, dropout_p=dropout_1, dilation=d2)
        self.enc3 = ResidualBlockSE(base_channels*2, base_channels*4, pool_type=pool_type, dropout_p=dropout_1, dilation=d3)
        self.enc4 = ResidualBlockSE(base_channels*4, base_channels*8, pool_type=pool_type, dropout_p=dropout_1, dilation=d4)
        self.pool = nn.MaxPool2d(2)

        # Optional FC bottleneck
        if use_fc_bottleneck:
            self.fc1 = nn.Linear(base_channels*8*fc_spatial*fc_spatial, fc_hidden)
            self.fc2 = nn.Linear(fc_hidden, base_channels*8*fc_spatial*fc_spatial)

        # Learned upsampling via ConvTranspose2d

        if self.predict_scalars:
            in_dim = fc_hidden if use_fc_bottleneck else base_channels*8
            self.fc_out = nn.Sequential(nn.Linear(in_dim,in_dim),nn.Linear(in_dim, self.n_scalars))


        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))
        self.loss_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.25, 0.125, 1,1,1,1], dtype=torch.float32))


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
            z = F.dropout(z, p=self.dropout_2, training=self.training)
            z = F.relu(self.fc2(z))
            z = F.dropout(z, p=self.dropout_2, training=self.training)
            e4 = F.interpolate(z.view(B, C, self.fc_spatial, self.fc_spatial),
                               size=(H, W), mode='bilinear', align_corners=False)
            feat = F.relu(self.fc1(
                F.adaptive_avg_pool2d(e4, (self.fc_spatial, self.fc_spatial)).view(B, -1)
            ))
        else:
            # no bottleneck: global pool + flatten
            B, C, H, W = e4.shape
            feat = F.adaptive_avg_pool2d(e4, 1).view(B, -1)

        if self.predict_scalars:
            # Return [B, n_scalars] instead of images
            return self.fc_out(feat)




    def criterion1(self,preds,target):
        return {'mse':F.mse_loss(preds, target)}
    def criterion(self, preds, target):
        losses = self.criterion1(preds,target)

        return sum(losses.values())

