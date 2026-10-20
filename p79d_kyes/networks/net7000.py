import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
import time
import datetime
import tqdm
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
import networks.net0600e as base_net
from networks.net0600e import SphericalDataset, d4_spin2, d4_spin2_inverse, grad_pseudo_stokes, rk_loss, set_seed
from networks.net0600d import pearson_loss, power_spectra_crit

idd = "7000"
what = "Residual conditional flow matching on top of frozen net0600e (CorrDiff-style). Samples Q,U given T."

# The regression model that supplies the conditional mean.  Its weights are loaded once
# in thisnet() and then stored inside this model's state_dict.
base_idd = "0600e"

fname_train = base_net.fname_train
fname_test = base_net.fname_test
load_data = base_net.load_data

device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 100
lr = 2e-4
batch_size = 32
weight_decay = 1e-4
ema_decay = 0.999
n_steps = 25          # Heun steps at sampling time (2 network evaluations each)
use_amp = True        # bf16 autocast on Ampere or newer, fp16 + loss scaling on older GPUs
use_checkpoint = True # recompute denoiser blocks in backward to save activation memory

def amp_dtype():
    if not (use_amp and torch.cuda.is_available()):
        return None
    # is_bf16_supported() is True on pre-Ampere cards through emulation, so check the hardware
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def thisnet():
    base = base_net.thisnet()
    base.load_state_dict(torch.load("models/test%s.pth" % base_idd, map_location=device))
    model = main_net(base)
    return model.to(device)

def train(model, all_data):
    trainer(model, all_data, epochs=epochs, lr=lr, batch_size=batch_size, weight_decay=weight_decay)

# ---------------------------
# Denoiser: periodic U-Net with time conditioning (FiLM) and GroupNorm
# ---------------------------
def gn(c):
    return nn.GroupNorm(min(32, c // 4), c)

def conv3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, padding_mode='circular')

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim*2), nn.SiLU(), nn.Linear(dim*2, dim*2))
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = 1000 * t[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=1))

class ResBlockT(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.norm1 = gn(cin)
        self.conv1 = conv3(cin, cout)
        self.film = nn.Linear(tdim, 2*cout)
        self.norm2 = gn(cout)
        self.conv2 = conv3(cout, cout)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(temb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)

class Denoiser(nn.Module):
    """v_theta(x_t, t, cond).  x_t: [B,2,H,W] noisy residual, cond: [B,Cc,H,W]."""
    def __init__(self, cond_channels, base_channels=64, mults=(1, 2, 4, 4), n_attn_layers=2, attn_heads=8):
        super().__init__()
        tdim = base_channels * 4
        self.temb = TimeEmbedding(tdim // 2)
        chans = [base_channels * m for m in mults]
        self.inp = conv3(2 + cond_channels, chans[0])
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        c = chans[0]
        for i, ch in enumerate(chans):
            self.down_blocks.append(ResBlockT(c, ch, tdim))
            c = ch
            if i < len(chans) - 1:
                self.downsamplers.append(conv3(c, c, stride=2))
        layer = nn.TransformerEncoderLayer(c, attn_heads, dim_feedforward=2*c, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, n_attn_layers, enable_nested_tensor=False)
        self.mid = ResBlockT(c, c, tdim)
        self.up_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for i in reversed(range(len(chans) - 1)):
            self.upconvs.append(conv3(c, c))
            self.up_blocks.append(ResBlockT(c + chans[i], chans[i], tdim))
            c = chans[i]
        self.out = nn.Sequential(gn(c), nn.SiLU(), conv3(c, 2))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def run(self, fn, *args):
        if use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x_t, t, cond):
        temb = self.temb(t)
        h = self.inp(torch.cat([x_t, cond], dim=1))
        skips = []
        for i, block in enumerate(self.down_blocks):
            h = self.run(block, h, temb)
            if i < len(self.downsamplers):
                skips.append(h)
                h = self.downsamplers[i](h)
        B, C, H, W = h.shape
        h = self.attn(h.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)
        h = self.mid(h, temb)
        for upconv, block in zip(self.upconvs, self.up_blocks):
            h = self.run(lambda h, s, temb, upconv=upconv, block=block:
                         block(torch.cat([upconv(F.interpolate(h, scale_factor=2, mode='nearest')), s], dim=1), temb),
                         h, skips.pop(), temb)
        return self.out(h)

# ---------------------------
# Full model
# ---------------------------
class main_net(nn.Module):
    """
    Q,U = mu(T) + T * sigma_r * x1,   x1 ~ p(x1 | T)
    mu is the frozen net0600e prediction.  x1 is generated by rectified flow matching
    from x0 ~ N(0,1): x_t = (1-t) x0 + t x1, the network learns v = x1 - x0.
    The residual is divided by T because net0600e predicts Q = T q, so the residual
    in q is closer to stationary.
    """
    def __init__(self, base, base_channels=64, epochs=epochs, n_steps=n_steps):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.n_steps = n_steps
        self.rotation_prob = 0
        self.denoiser = Denoiser(cond_channels=6, base_channels=base_channels)
        self.ema = copy.deepcopy(self.denoiser)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.register_buffer("sigma_r", torch.ones(2))
        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()          # the frozen regression model always stays in eval mode
        self.ema.eval()
        return self

    @torch.no_grad()
    def condition(self, T, tta=False):
        """T [B,1,H,W] -> (mu [B,2,H,W] in Q,U units, cond [B,6,H,W])"""
        tta_save, self.base.tta = self.base.tta, tta
        mu = self.base(T)[0][:, 1:]
        self.base.tta = tta_save
        logT = torch.log(T.clamp(min=1e-3))
        gq, gu, g2 = grad_pseudo_stokes(logT)
        feats = self.base.in_norm(torch.cat([logT, gq, gu, g2], dim=1))
        cond = torch.cat([feats, mu / T], dim=1)
        return mu, cond

    def to_x1(self, QU, mu, T):
        return (QU - mu) / T / self.sigma_r[None, :, None, None]

    def from_x1(self, x1, mu, T):
        return mu + T * self.sigma_r[None, :, None, None] * x1

    def flow_loss(self, T, QU):
        if T.ndim == 3:
            T = T.unsqueeze(1)
        mu, cond = self.condition(T)
        x1 = self.to_x1(QU, mu, T)
        x0 = torch.randn_like(x1)
        t = torch.sigmoid(torch.randn(T.shape[0], device=T.device))   # logit-normal times
        tt = t[:, None, None, None]
        x_t = (1 - tt) * x0 + tt * x1
        with torch.autocast('cuda', dtype=amp_dtype() or torch.float32, enabled=amp_dtype() is not None):
            v = self.denoiser(x_t, t, cond)
        return F.mse_loss(v.float(), x1 - x0)

    @torch.no_grad()
    def sample(self, T, n_samples=1, n_steps=None, tta_mean=False):
        """T [B,H,W] or [B,1,H,W] -> samples [n_samples, B, 3, H, W] (channels T, Q, U)"""
        if T.ndim == 3:
            T = T.unsqueeze(1)
        n_steps = n_steps or self.n_steps
        mu, cond = self.condition(T, tta=tta_mean)
        net = self.ema
        out = []
        for _ in range(n_samples):
            x = torch.randn(T.shape[0], 2, *T.shape[-2:], device=T.device)
            ts = torch.linspace(0, 1, n_steps + 1, device=T.device)
            for i in range(n_steps):
                t0, t1 = ts[i], ts[i+1]
                with torch.autocast('cuda', dtype=amp_dtype() or torch.float32, enabled=amp_dtype() is not None):
                    v0 = net(x, t0.expand(T.shape[0]), cond).float()
                    x_pred = x + (t1 - t0) * v0
                    v1 = net(x_pred, t1.expand(T.shape[0]), cond).float()
                x = x + (t1 - t0) * 0.5 * (v0 + v1)
            out.append(torch.cat([T, self.from_x1(x, mu, T)], dim=1))
        return torch.stack(out)

    def forward(self, x):
        """Inference only (training goes through flow_loss).  Returns (one sample, regression mean),
        both [B,3,H,W], so plotter.plot1 shows a single draw in its spectra columns."""
        if x.ndim == 3:
            x = x.unsqueeze(1)
        sample = self.sample(x, n_samples=1)[0]
        mu, _ = self.condition(x, tta=True)
        return sample, torch.cat([x, mu], dim=1)

    def criterion1(self, preds, target):
        """Diagnostics for plotter.plot1, on the single sample in preds[0]."""
        p, t = preds[0][:, 1:], target[:, 1:]
        return {'L1_0': F.l1_loss(p, t),
                'L1_mean': F.l1_loss(preds[1][:, 1:], t),
                'Pear': (pearson_loss(p[:, 0:1], t[:, 0:1]) + pearson_loss(p[:, 1:2], t[:, 1:2])) / 2,
                'Power': power_spectra_crit(preds[0], target)}

    def criterion(self, preds, target):
        return sum(self.criterion1(preds, target).values())

    @torch.no_grad()
    def update_ema(self, decay=ema_decay):
        for pe, p in zip(self.ema.parameters(), self.denoiser.parameters()):
            pe.lerp_(p, 1 - decay)

@torch.no_grad()
def estimate_sigma_r(model, ds, nmaps=512):
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    acc, n = 0, 0
    for xb, yb in loader:
        T = xb.to(device).unsqueeze(1)
        mu, _ = model.condition(T)
        r = (yb.to(device)[:, 1:] - mu) / T
        acc = acc + (r**2).mean((0, 2, 3)) * xb.shape[0]
        n += xb.shape[0]
        if n >= nmaps:
            break
    return torch.sqrt(acc / n)

def trainer(model, all_data, epochs=100, batch_size=32, lr=2e-4, weight_decay=1e-4, grad_clip=1.0):
    set_seed()
    ds_train = SphericalDataset(all_data['train'], augment=True)
    ds_val = SphericalDataset(all_data['valid'], augment=False)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.sigma_r.copy_(estimate_sigma_r(model, SphericalDataset(all_data['train'], augment=False)))
    print("residual scale sigma_r", model.sigma_r.tolist(), "| autocast", amp_dtype(), "| checkpoint", use_checkpoint)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype() == torch.float16)
    model.ema.load_state_dict(model.denoiser.state_dict())

    optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.02)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in tqdm.tqdm(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.flow_loss(xb, yb[:, 1:])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.update_ema()
            running += loss.item()
        train_loss = running / len(train_loader)

        # validation flow loss with the EMA weights, fixed noise so epochs are comparable
        model.eval()
        live, model.denoiser = model.denoiser, model.ema
        rng = torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(1234)
        with torch.no_grad():
            vtotal = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vtotal += model.flow_loss(xb, yb[:, 1:]).item() * xb.shape[0]
            val_loss = vtotal / len(ds_val)
        torch.set_rng_state(rng[0])
        if rng[1] is not None:
            torch.cuda.set_rng_state_all(rng[1])
        model.denoiser = live
        model.train_curve[epoch-1] = train_loss
        model.val_curve[epoch-1] = val_loss

        now = time.time()
        secs_left = (now - t0) / epoch * (epochs - epoch)
        eta = datetime.datetime.fromtimestamp(now + secs_left).strftime("%H:%M:%S")
        print(f"[{epoch:3d}/{epochs}] net{idd}  train {train_loss:.4f} | val(EMA) {val_loss:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | ETA {eta}")
    return model
