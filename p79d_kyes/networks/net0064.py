# GPU-friendly training boilerplate (drop-in replacement parts)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import time
import datetime
import os

# --- Config / device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Tune these to your machine / GPU
DEFAULT_BATCH_SIZE = 16    # try increasing until you hit memory limit
DEFAULT_NUM_WORKERS = min(8, max(1, (os.cpu_count() or 4) // 2))
PIN_MEMORY = True

# --- Determinism / cudnn ---
def set_seed(seed=8675309):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# If your input sizes (H,W) are constant or mostly constant, this helps:
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# --- downsample util (works for numpy arrays or torch tensors) ---
def downsample_avg(x, M):
    """
    x: numpy array with shape [B, C, H, W] or torch tensor same shape OR single image [H,W]/[C,H,W]
    M: target spatial size
    returns: torch.FloatTensor [B, C, M, M] or [C, M, M] or [M, M]
    """
    # convert to tensor if necessary
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.ndim == 2:   # [N, N]
        x = x.unsqueeze(0).unsqueeze(0).float()  # -> [1, 1, N, N]
        out = F.adaptive_avg_pool2d(x, (M, M))
        return out.squeeze(0).squeeze(0) # -> [M, M]
    elif x.ndim == 3:  # [C, H, W]
        x = x.unsqueeze(0).float()  # -> [1, C, H, W]
        out = F.adaptive_avg_pool2d(x, (M, M))
        return out.squeeze(0)       # -> [C, M, M]
    elif x.ndim == 4: # [B, C, N, N]
        x = x.float()
        return F.adaptive_avg_pool2d(x, (M, M))
    else:
        raise ValueError("Input must be [N, N], [C, N, N] or [B, C, N, N]")

# ---------------------------
# Dataset with input normalization (converts to float32 torch tensors once)
# ---------------------------
from torch.utils.data import Dataset, DataLoader

class SphericalDataset(Dataset):
    def __init__(self, all_data, downsample_size=None):
        """
        all_data expected shape: [B, ?] where each element is (input, targets...) OR a tensor of shape [B, C+?, H, W]
        Adapted to your loader output: all_data['train'] appears to be an array-like where each element has [input, outputs...].
        """
        # If loader returns a numpy array of shape [B, 3, H, W] (example), adapt accordingly.
        # Here I support both: 1) a torch/numpy array [B, C_total, H, W]
        #                     2) a list/array-like where each item is a tuple (inp, targets...) or small array
        self.downsample_size = downsample_size
        self._data = []

        # handle numpy/torch array case
        if isinstance(all_data, (np.ndarray, torch.Tensor)):
            arr = torch.from_numpy(all_data) if isinstance(all_data, np.ndarray) else all_data
            arr = arr.float()
            if self.downsample_size:
                arr = downsample_avg(arr, self.downsample_size)
            # assume arr shape [B, C_total, H, W]; first channel(s) input, rest targets
            self._arr = arr
            self._use_arr = True
        else:
            # assume iterable of pairs: each element is indexable (inp, out1, out2...)
            for item in all_data:
                # convert each component to float tensor
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    inp = item[0]
                    outs = item[1:]
                    # convert
                    if isinstance(inp, np.ndarray):
                        inp = torch.from_numpy(inp).float()
                    else:
                        inp = torch.as_tensor(inp).float()
                    outs_t = []
                    for o in outs:
                        if isinstance(o, np.ndarray):
                            outs_t.append(torch.from_numpy(o).float())
                        else:
                            outs_t.append(torch.as_tensor(o).float())
                    # stack targets along channel dim if shape allows
                    try:
                        # ensure shapes are [C,H,W] or [H,W]
                        self._data.append((inp, torch.stack(outs_t, dim=0)))
                    except Exception:
                        # fallback: keep as tuple
                        self._data.append((inp, tuple(outs_t)))
            self._use_arr = False

    def __len__(self):
        if self._use_arr:
            return self._arr.size(0)
        else:
            return len(self._data)

    def __getitem__(self, idx):
        if self._use_arr:
            # split channels: assume input is first channel, targets the rest
            x = self._arr[idx:idx+1, 0:1] if self._arr.ndim==4 else self._arr[idx, 0:1]
            y = self._arr[idx, 1:, ...] if self._arr.ndim==4 else self._arr[idx, 1:, ...]
            return x, y
        else:
            x, y = self._data[idx]
            # if x is [H,W], make channel dim
            if x.ndim == 2:
                x = x.unsqueeze(0)
            return x, y

# ---------------------------
# Trainer with AMP, pin_memory, non_blocking transfers
# ---------------------------
from torch.cuda.amp import GradScaler, autocast

def trainer(
    model,
    all_data,
    epochs=200,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=1e-4,
    weight_decay=1e-4,
    grad_clip=None,
    warmup_frac=0.05,
    device=device,
    lr_schedule=[900],
    plot_path=None,
    num_workers=DEFAULT_NUM_WORKERS,
    pin_memory=PIN_MEMORY,
):
    set_seed()
    model = model.to(device)

    # build datasets
    ds_train = SphericalDataset(all_data['train'], downsample_size=downsample if downsample else None)
    ds_val   = SphericalDataset(all_data['valid'], downsample_size=downsample if downsample else None)

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers>0),
        drop_last=False,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=max(64, batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers>0),
        drop_last=False,
        prefetch_factor=2
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_schedule, gamma=0.1)

    scaler = GradScaler(enabled=(torch.cuda.is_available()))
    best_val = float("inf")
    best_state = None
    patience = 25
    bad_epochs = 0

    train_curve, val_curve = [], []
    t0 = time.time()

    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            # ensure tensors are float and have channel dims
            xb = xb.float()
            yb = yb.float()

            # move to device (non_blocking if pin_memory=True)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(torch.cuda.is_available())):
                preds = model(xb)
                loss = model.criterion(preds, yb)

            # scale & backward
            scaler.scale(loss).backward()

            # optional gradient clipping (unscale first)
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * xb.size(0)

        scheduler.step()
        train_loss = running / len(ds_train)
        train_curve.append(train_loss)

        # validation
        model.eval()
        with torch.no_grad():
            vtotal = 0.0
            for xb, yb in val_loader:
                xb = xb.float().to(device, non_blocking=True)
                yb = yb.float().to(device, non_blocking=True)
                with autocast(enabled=(torch.cuda.is_available())):
                    preds = model(xb)
                    vloss = model.criterion(preds, yb)
                vtotal += vloss.item() * xb.size(0)
            val_loss = vtotal / len(ds_val)
            val_curve.append(val_loss)

        # early stopping logic
        improved = val_loss < best_val - 1e-7
        if improved:
            best_val = val_loss
            # copy CPU weights for safety
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        # logging
        now = time.time()
        time_per_epoch = (now - t0) / epoch
        secs_left = time_per_epoch * (epochs - epoch)
        etad = datetime.datetime.fromtimestamp(now + secs_left)
        eta = etad.strftime("%Y-%m-%d %H:%M:%S")
        lr_now = optimizer.param_groups[0]['lr']

        def format_time(seconds):
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        elps = format_time(now - t0)
        rem  = format_time(secs_left if secs_left>0 else 0)

        model.train_curve = torch.tensor(train_curve)
        model.val_curve = torch.tensor(val_curve)

        print(f"[{epoch:3d}/{epochs}] train {train_loss:.6f} | val {val_loss:.6f} | lr {lr_now:.2e} | bad {bad_epochs:02d} | ETA {eta} | Remain {rem} | SoFar {elps}")

        # optional early stopping
        #if bad_epochs >= patience:
        #    print(f"Stopped early at epoch {epoch}, best val {best_val:.6f}")
        #    break

    # restore best state if available
    #if best_state is not None:
    #    model.load_state_dict(best_state)
    return model

# ---------------------------
# Example usage (wrap your thisnet/train functions)
# ---------------------------
def thisnet():
    model = main_net(base_channels=32, fc_spatial=4, use_fc_bottleneck=True)
    # initialize weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model = model.to(device)
    return model

def train(model, all_data):
    # pick a batch size that fits your GPU; increase batch_size for better utilization
    batch_size = 16
    epochs = 500
    lr = 1e-3
    lr_schedule = [100]
    trainer(model, all_data, epochs=epochs, lr=lr, batch_size=batch_size, weight_decay=0, lr_schedule=lr_schedule, num_workers=DEFAULT_NUM_WORKERS)

