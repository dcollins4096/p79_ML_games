import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import datetime
import tqdm
from torch.utils.data import Dataset, DataLoader
import loader
import networks.net0600d as base
from networks.net0600d import ResidualBlockSE, ssim_loss, gradient_loss, pearson_loss, set_seed

idd = "0600f"
what = "0600e + velocity centroid and variance inputs. 0600d + periodic convs, D4/roll spin-2 aug, gradient pseudo-Stokes inputs, Q=T*q, attention bottleneck, mean head, r(k) loss, TTA"

fname_train = "p79d_dataset_S512_N1_xyz_down_128_brano_tvsquhp_NOT4.h5"
fname_test  = "p79d_dataset_S512_N1_xyz_down_128_brano_tvsquhp_YES4.h5"

# Validation is a held-out sonic Mach number (all three Ma, all frames), so it measures
# generalization the same way the Mach 4 test does.  Set to None for the old random 30.
valid_ms_range = (7.0, 8.2)
nvalid_old = 30
valid_stride = 3        # use every 3rd held-out map to keep validation cheap

device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 50
lr = 1e-3
batch_size = 32   # 64 runs out of memory on the 7.6 GB card (circular padding copies each activation)
weight_decay = 1e-3

# Dataset channels used as input: T, velocity centroid, velocity variance.  Targets are always T, Q, U.
input_channels = [0, 1, 2]
v_scale = 3e-3        # typical centroid std (0.001 at Ms~1 to 0.007 at Ms~10); fixed, so amplitude still tracks Ms

def load_data():
    print('read the data')
    tr = loader.loader2(fname_train, ntest=0, nvalid=1)
    subsets = torch.cat([tr['train'], tr['valid']])
    quan = {q: np.concatenate([tr['quantities']['train'][q], tr['quantities']['valid'][q]]) for q in tr['quantities']['train']}
    if valid_ms_range is not None:
        ms = quan['Ms_mean']
        vmask = (ms > valid_ms_range[0]) & (ms < valid_ms_range[1])
        vidx = np.where(vmask)[0][::valid_stride]
        tidx = np.where(~vmask)[0]
    else:
        vidx = np.arange(len(subsets))[-nvalid_old:]
        tidx = np.arange(len(subsets))[:-nvalid_old]
    te = loader.loader2(fname_test, ntest=10000, nvalid=1)
    all_data = {'train': subsets[tidx], 'valid': subsets[vidx], 'test': te['test'], 'quantities': {}}
    all_data['quantities']['train'] = {q: v[tidx] for q, v in quan.items()}
    all_data['quantities']['valid'] = {q: v[vidx] for q, v in quan.items()}
    all_data['quantities']['test'] = te['quantities']['test']
    print('done. train %d valid %d test %d' % (len(tidx), len(vidx), len(all_data['test'])))
    return all_data

def thisnet():
    model = main_net(base_channels=32, out_channels=3)
    model = model.to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model

def train(model, all_data):
    trainer(model, all_data, epochs=epochs, lr=lr, batch_size=batch_size, weight_decay=weight_decay)

# ---------------------------
# Symmetries.  The maps are periodic, and Q,U are spin-2:
#   rot90 (either direction):  Q -> -Q, U -> -U
#   flip of one axis:          Q ->  Q, U -> -U
# T, the velocity centroid and the velocity variance are scalars.  x is [..., C, H, W]
# with Q, U as the LAST two channels; all channels before them are left alone.
# ---------------------------
def d4_spin2(x, k, flip):
    x = x.clone()
    if flip:
        x = x.flip(-1)
        x[..., -1, :, :] *= -1
    if k % 4:
        x = torch.rot90(x, k, dims=(-2, -1))
        if k % 2:
            x[..., -2:, :, :] *= -1
    return x

def d4_spin2_inverse(x, k, flip):
    x = x.clone()
    if k % 4:
        x = torch.rot90(x, -k, dims=(-2, -1))
        if k % 2:
            x[..., -2:, :, :] *= -1
    if flip:
        x = x.flip(-1)
        x[..., -1, :, :] *= -1
    return x

class SphericalDataset(Dataset):
    def __init__(self, all_data, augment=False):
        self.all_data = all_data
        self.augment = augment
    def __len__(self):
        return self.all_data.size(0)
    def __getitem__(self, idx):
        theset = self.all_data[idx][input_channels + [3, 4]]
        if self.augment:
            theset = d4_spin2(theset, random.randint(0, 3), random.random() < 0.5)
            theset = torch.roll(theset, (random.randrange(theset.shape[-2]), random.randrange(theset.shape[-1])), dims=(-2, -1))
        return theset[:-2], theset[[0, -2, -1]]

# ---------------------------
# Losses
# ---------------------------
def rk_loss(pred, target, nbins=12, eps=1e-12):
    """1 - r(k), with r(k) the batch-aggregated cross-correlation coefficient in
    log-spaced k bins, averaged over bins.  Rewards getting the phases right at every
    scale, but (unlike a power-spectrum loss) does not reward hallucinated power."""
    N = pred.shape[-1]
    P = torch.fft.fft2(pred)
    T = torch.fft.fft2(target)
    kx = torch.fft.fftfreq(N, device=pred.device) * N
    k = torch.sqrt(kx[:, None]**2 + kx[None, :]**2)
    edges = torch.logspace(0, np.log10(N / 2), nbins + 1, device=pred.device)
    idx = (torch.bucketize(k, edges) - 1).clamp(-1, nbins - 1)
    keep = idx >= 0                     # drop k=0; the mean has its own loss
    idx = idx[keep]
    cross = (P * T.conj()).real.sum(0)[..., keep]   # [C, npix]
    pp = (P.abs()**2).sum(0)[..., keep]
    tt = (T.abs()**2).sum(0)[..., keep]
    C = cross.shape[0]
    def binsum(a):
        return torch.zeros(C, nbins, device=a.device).index_add_(1, idx, a)
    r = binsum(cross) / torch.sqrt(binsum(pp) * binsum(tt) + eps)
    return 1 - r.mean()

def grad_pseudo_stokes(logT):
    """Density-gradient pseudo-Stokes parameters (spin-2 like Q,U) on a periodic map."""
    tx = 0.5 * (torch.roll(logT, -1, -1) - torch.roll(logT, 1, -1))
    ty = 0.5 * (torch.roll(logT, -1, -2) - torch.roll(logT, 1, -2))
    return tx**2 - ty**2, 2 * tx * ty, tx**2 + ty**2

def make_periodic(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.padding != (0, 0):
            m.padding_mode = 'circular'
    return module

class UpConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1, padding_mode='circular')
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))

class main_net(nn.Module):
    def __init__(self, out_channels=3, base_channels=32, pool_type='max', epochs=epochs,
                 n_attn_layers=2, attn_heads=8, multiply_T=True, tta=True,
                 err_L1=1, err_Multi=1, err_Pear=1, err_SSIM=1, err_Grad=1, err_Rk=1, err_Mean=1,
                 err_Power=0, err_Cross=0):
        super().__init__()
        self.multiply_T = multiply_T
        self.tta = tta
        self.err_L1 = err_L1
        self.err_Multi = err_Multi
        self.err_Pear = err_Pear
        self.err_SSIM = err_SSIM
        self.err_Grad = err_Grad
        self.err_Rk = err_Rk
        self.err_Mean = err_Mean
        self.err_Power = err_Power
        self.err_Cross = err_Cross
        self.rotation_prob = 0
        bc = base_channels

        # log T, its gradient pseudo-Stokes (3), v, log variance, v gradient pseudo-Stokes (3)
        in_channels = 9
        self.in_norm = nn.BatchNorm2d(in_channels)

        self.enc1 = ResidualBlockSE(in_channels, bc, pool_type=pool_type)
        self.enc2 = ResidualBlockSE(bc, bc*2, pool_type=pool_type)
        self.enc3 = ResidualBlockSE(bc*2, bc*4, pool_type=pool_type)
        self.enc4 = ResidualBlockSE(bc*4, bc*8, pool_type=pool_type)
        self.pool = nn.MaxPool2d(2)

        # Global context: self-attention over the 16x16 bottleneck.  No positional
        # encoding, so it stays translation-equivariant on the periodic box.
        layer = nn.TransformerEncoderLayer(bc*8, attn_heads, dim_feedforward=bc*16,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, n_attn_layers)

        # Global head: map-mean of q,u (the k=0 mode carried ~half the error in 0600d)
        self.mean_head = nn.Sequential(nn.Linear(bc*8*2, 256), nn.ReLU(), nn.Linear(256, 2))

        self.up4 = UpConv(bc*8)
        self.up3 = UpConv(bc*4)
        self.up2 = UpConv(bc*2)
        self.dec4 = ResidualBlockSE(bc*8 + bc*4, bc*4, pool_type=pool_type)
        self.dec3 = ResidualBlockSE(bc*4 + bc*2, bc*2, pool_type=pool_type)
        self.dec2 = ResidualBlockSE(bc*2 + bc, bc, pool_type=pool_type)
        self.dec1 = nn.Conv2d(bc, 2, 3, padding=1)
        self.out_d4 = nn.Conv2d(bc*4, 2, 3, padding=1)
        self.out_d3 = nn.Conv2d(bc*2, 2, 3, padding=1)
        self.out_d2 = nn.Conv2d(bc, 2, 3, padding=1)
        make_periodic(self)

        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))

    def _assemble(self, T, qu, qu_mean):
        """T [B,1,h,w], qu [B,2,h,w] -> [B,3,h,w] = T, Q, U"""
        if self.multiply_T:
            QU = T * (qu + qu_mean[:, :, None, None])
        else:
            QU = qu + qu_mean[:, :, None, None]
        return torch.cat([T, QU], dim=1)

    def input_features(self, x):
        """x [B,3,H,W] = T, velocity centroid, velocity variance -> normalized [B,9,H,W]"""
        T, v, var = x[:, 0:1], x[:, 1:2] / v_scale, x[:, 2:3]
        logT = torch.log(T.clamp(min=1e-3))
        gq, gu, g2 = grad_pseudo_stokes(logT)
        vq, vu, v2 = grad_pseudo_stokes(v)
        logvar = torch.log(var.clamp(min=1e-6))
        return self.in_norm(torch.cat([logT, gq, gu, g2, v, logvar, vq, vu, v2], dim=1))

    def forward1(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        T = x[:, 0:1]
        h = self.input_features(x)

        e1 = self.enc1(h)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        B, C, H, W = e4.shape
        tok = self.attn(e4.flatten(2).transpose(1, 2))          # [B, HW, C]
        e4 = tok.transpose(1, 2).reshape(B, C, H, W)
        qu_mean = self.mean_head(torch.cat([tok.mean(1), tok.amax(1)], dim=1))

        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))

        out_main = self._assemble(T, self.dec1(d2), qu_mean)
        out_d2 = self._assemble(T, self.out_d2(d2), qu_mean)
        out_d3 = self._assemble(F.avg_pool2d(T, 2), self.out_d3(d3), qu_mean)
        out_d4 = self._assemble(F.avg_pool2d(T, 4), self.out_d4(d4), qu_mean)
        return out_main, out_d2, out_d3, out_d4

    def forward(self, x):
        preds = self.forward1(x)
        if self.training or not self.tta:
            return preds
        # Test-time augmentation: average the main output over the 8 D4 transforms.
        if x.ndim == 3:
            x = x.unsqueeze(1)
        acc = 0
        for k in range(4):
            for flip in (False, True):
                xx = torch.rot90(x.flip(-1) if flip else x, k, dims=(-2, -1))
                acc = acc + d4_spin2_inverse(self.forward1(xx)[0], k, flip)
        return (acc / 8,) + preds[1:]

    def criterion1(self, preds, target):
        out_main, out_d2, out_d3, out_d4 = preds
        # T is copied from the input, so only Q,U carry loss.
        p, t = out_main[:, 1:], target[:, 1:]
        all_loss = {}
        if self.err_L1 > 0:
            all_loss['L1_0'] = self.err_L1 * F.l1_loss(p, t)
        if self.err_Multi > 0:
            lm = 0
            for o in (out_d2, out_d3, out_d4):
                lm = lm + F.l1_loss(o[:, 1:], F.adaptive_avg_pool2d(t, o.shape[-2:]))
            all_loss['L1_Multi'] = self.err_Multi * lm
        if self.err_Mean > 0:
            all_loss['Mean'] = self.err_Mean * F.l1_loss(p.mean((-2, -1)), t.mean((-2, -1)))
        if self.err_SSIM > 0:
            all_loss['SSIM'] = self.err_SSIM * (ssim_loss(p[:, 0:1], t[:, 0:1]) + ssim_loss(p[:, 1:2], t[:, 1:2])) / 2
        if self.err_Grad > 0:
            all_loss['Grad'] = self.err_Grad * gradient_loss(p, t)
        if self.err_Pear > 0:
            all_loss['Pear'] = self.err_Pear * (pearson_loss(p[:, 0:1], t[:, 0:1]) + pearson_loss(p[:, 1:2], t[:, 1:2])) / 2
        if self.err_Rk > 0:
            all_loss['Rk'] = self.err_Rk * rk_loss(p, t)
        if self.err_Power > 0:
            all_loss['Power'] = self.err_Power * base.power_spectra_crit(out_main, target)
        if self.err_Cross > 0:
            all_loss['Cross'] = self.err_Cross * base.cross_spectra_crit(out_main, target)
        return all_loss

    def criterion(self, preds, target):
        return sum(self.criterion1(preds, target).values())

def trainer(model, all_data, epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-3, grad_clip=1.0):
    set_seed()
    ds_train = SphericalDataset(all_data['train'], augment=True)
    ds_val = SphericalDataset(all_data['valid'], augment=False)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.05)

    best_val, best_state = float('inf'), None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in tqdm.tqdm(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            running += loss.item() * xb.size(0)
        train_loss = running / (len(train_loader) * batch_size)

        model.eval()
        tta, model.tta = model.tta, False
        with torch.no_grad():
            vtotal = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vtotal += model.criterion(model(xb), yb).item() * xb.size(0)
            val_loss = vtotal / len(ds_val)
        model.tta = tta
        model.train_curve[epoch-1] = train_loss
        model.val_curve[epoch-1] = val_loss

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        now = time.time()
        secs_left = (now - t0) / epoch * (epochs - epoch)
        eta = datetime.datetime.fromtimestamp(now + secs_left).strftime("%H:%M:%S")
        print(f"[{epoch:3d}/{epochs}] net{idd}  train {train_loss:.4f} | val {val_loss:.4f} | best {best_val:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | ETA {eta}")

    # restore the best held-out-Mach model (curves are kept from the final epoch)
    curves = model.train_curve.clone(), model.val_curve.clone()
    model.load_state_dict(best_state)
    model.train_curve.copy_(curves[0])
    model.val_curve.copy_(curves[1])
    return model
