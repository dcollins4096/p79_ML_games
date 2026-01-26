import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
import time
import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import loader

idd = 9004
what = "Hybrid CNN-ViT: CNN stem for local features + ViT for global context"

dirpath = "/home/dcollins/repos/p79_ML_games/p79d_kyes/datasets/"
dirpath = "./"

fname_train = "p79d_subsets_S128_N1_xyz_suite7vs_first.h5"
fname_valid = "p79d_subsets_S128_N1_xyz_suite7vs_second.h5"

ntrain = 14000
nvalid = 100
ntest = 5000
downsample = None
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Hyperparameters
epochs = 500
lr = 5e-4
batch_size = 64
lr_schedule = [1000]
weight_decay = 0.01

# Hybrid architecture parameters
img_size = 128
cnn_output_size = 64  # After CNN pooling: 128 → 64
patch_size = 8  # Smaller patches on reduced resolution
embed_dim = 384
depth = 6
num_heads = 6
mlp_ratio = 4.0
dropout = 0.1


def load_data():
    print('read the data')
    train_raw = loader.loader(dirpath + fname_train, ntrain=ntrain, nvalid=0)
    valid_raw = loader.loader(dirpath + fname_valid, ntrain=1, nvalid=nvalid)
    
    all_data_combined = train_raw['train']
    all_ms_combined = train_raw['quantities']['train']['Ms_act']
    
    all_test_data = valid_raw['test']
    all_test_ms = valid_raw['quantities']['test']['Ms_act']
    
    # Stratified validation
    mach_bins = [0, 4, 6, 8, 10, 15]
    samples_per_bin = nvalid // (len(mach_bins) - 1)
    
    valid_indices = []
    for i in range(len(mach_bins) - 1):
        mask = (all_ms_combined >= mach_bins[i]) & (all_ms_combined < mach_bins[i+1])
        bin_indices = torch.where(torch.from_numpy(mask))[0].cpu()
        
        if len(bin_indices) >= samples_per_bin:
            selected = bin_indices[torch.randperm(len(bin_indices))[:samples_per_bin]]
            valid_indices.extend(selected.tolist())
        else:
            valid_indices.extend(bin_indices.tolist())
            print(f"Warning: Only {len(bin_indices)} samples in Ms range [{mach_bins[i]}, {mach_bins[i+1]})")
    
    valid_indices = torch.tensor(valid_indices)
    
    valid_data = all_data_combined[valid_indices]
    valid_ms = all_ms_combined[valid_indices]
    
    all_indices = torch.arange(len(all_data_combined))
    train_mask = ~torch.isin(all_indices, valid_indices)
    train_data = all_data_combined[train_mask]
    train_ms = all_ms_combined[train_mask]
    
    all_data = {
        'train': train_data,
        'valid': valid_data,
        'test': all_test_data[:ntest],
        'quantities': {}
    }
    
    all_data['quantities']['train'] = {
        'Ms_act': train_ms,
        'Ma_act': train_raw['quantities']['train']['Ma_act'][train_mask]
    }
    all_data['quantities']['valid'] = {
        'Ms_act': valid_ms,
        'Ma_act': train_raw['quantities']['train']['Ma_act'][valid_indices]
    }
    all_data['quantities']['test'] = {
        'Ms_act': all_test_ms[:ntest],
        'Ma_act': train_raw['quantities']['train']['Ma_act'][:ntest]
    }
    
    print(f'Train: {len(train_data)}, Valid (stratified): {len(valid_data)}, Test: {min(ntest, len(all_test_data))}')
    
    print('\nValidation set Mach distribution:')
    for i in range(len(mach_bins)-1):
        mask = (valid_ms >= mach_bins[i]) & (valid_ms < mach_bins[i+1])
        print(f"  Ms [{mach_bins[i]:2d}-{mach_bins[i+1]:2d}): {mask.sum():3d} samples")
    
    print('done')
    return all_data


def thisnet():
    model = HybridCNNViT(
        img_size=img_size,
        cnn_output_size=cnn_output_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=1,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout
    )
    model = model.to(device)
    return model


def train(model, all_data):
    trainer(model, all_data, epochs=epochs, lr=lr, batch_size=batch_size,
            weight_decay=weight_decay, lr_schedule=lr_schedule)


class SphericalDataset(Dataset):
    def __init__(self, all_data, quan, augment=False):
        self.quan = quan
        self.all_data = all_data
        self.augment = augment

    def __len__(self):
        return self.all_data.size(0)

    def __getitem__(self, idx):
        data = self.all_data[idx][0:3]
        
        if self.augment:
            H, W = data[0].shape
            dy = torch.randint(0, H, (1,)).item()
            dx = torch.randint(0, W, (1,)).item()
            data = torch.roll(data, shifts=(dy, dx), dims=(-2, -1))
            
            if torch.rand(1) > 0.5:
                data = torch.flip(data, dims=[-1])
            if torch.rand(1) > 0.5:
                data = torch.flip(data, dims=[-2])
        
        ms = self.quan['Ms_act'][idx]
        #return data.to(device), torch.tensor([ms], dtype=torch.float32).to(device)
        return data.to(device), torch.log(torch.tensor([ms], dtype=torch.float32).to(device))
        

def set_seed(seed=8675309):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================
# CNN STEM - Extracts Local Features
# ==========================================
class CNNStem(nn.Module):
    """
    CNN stem to extract local turbulence features:
    - Density gradients
    - Velocity shear
    - Shock structures

    Reduces spatial resolution: 128x128 → 64x64
    Increases channels: 3 → 64
    """
    def __init__(self, in_chans=3, out_chans=64):
        super().__init__()
        
        # First conv block: capture local patterns
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second conv block: build hierarchical features
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third conv block: refine features before pooling
        self.conv3 = nn.Conv2d(64, out_chans, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_chans)
        
        # Pooling: reduce resolution for efficient attention
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # x: [B, 3, 128, 128]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 32, 128, 128]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 64, 128, 128]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 64, 128, 128]
        x = self.pool(x)                      # [B, 64, 64, 64]
        return x


# ==========================================
# ViT Components (same as before)
# ==========================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=64, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 64 patches for 64×64 with 8×8 patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ==========================================
# HYBRID CNN-ViT MODEL
# ==========================================
class HybridCNNViT(nn.Module):
    """
    Hybrid architecture combining CNNs and Vision Transformers:
    
    Stage 1 (CNN Stem):
        - Extract local turbulence features (gradients, shocks, shear)
        - Reduce resolution: 128x128 → 64x64
        - Increase channels: 3 → 64
        
    Stage 2 (Vision Transformer):
        - Process CNN features with global self-attention
        - Capture long-range flow structures and correlations
        - Make final Mach number prediction
    
    This combines CNN's local inductive bias with ViT's global context,
    proven effective for limited scientific data (Xiao et al., NeurIPS 2021)
    """
    def __init__(self, img_size=128, cnn_output_size=64, patch_size=8, 
                 in_chans=3, num_classes=1, embed_dim=384, depth=6, 
                 num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        # Stage 1: CNN stem for local features
        self.cnn_stem = CNNStem(in_chans=in_chans, out_chans=64)
        
        # Stage 2: Patch embedding on CNN features
        self.patch_embed = PatchEmbed(
            img_size=cnn_output_size,  # 64×64 after CNN
            patch_size=patch_size,      # 8×8 patches
            in_chans=64,                # CNN output channels
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.n_patches

        # ViT components
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Training curves
        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))
        self.predict_scalars=True
        self.n_scalars=1
        self.predict_log=True

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        
        # Stage 1: CNN extracts local features
        # [B, 3, 128, 128] → [B, 64, 64, 64]
        x = self.cnn_stem(x)
        
        # Stage 2: ViT processes CNN features
        # [B, 64, 64, 64] → [B, 64, 384] (64 patches, 384-dim embeddings)
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 65, 384]
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        
        # Use class token for prediction
        x = x[:, 0]
        x = self.head(x)

        return x

    def criterion1(self, preds, target):
        mse = F.mse_loss(preds, target)
        huber = F.smooth_l1_loss(preds, target)
        return {'mse': mse, 'huber': huber}

    def criterion(self, preds, target):
        losses = self.criterion1(preds, target)
        
        # Mild weighting for high Mach (Ms > 10)
        weights = torch.where(target > 10.0, 2.0, 1.0)
        weighted_mse = (weights * (preds - target)**2).mean()
        
        return 0.6 * losses['mse'] + 0.3 * losses['huber'] + 0.1 * weighted_mse


def trainer(model, all_data, epochs=50, batch_size=64, lr=5e-4,
            weight_decay=0.01, lr_schedule=[1000]):
    #set_seed()

    ds_train = SphericalDataset(all_data['train'], all_data['quantities']['train'], augment=True)
    ds_val = SphericalDataset(all_data['valid'], all_data['quantities']['valid'], augment=False)
    
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(ds_val, batch_size=max(64, batch_size), shuffle=False, drop_last=False)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    print("Total Steps", total_steps, "ntrain", len(ds_train), "epoch", epochs)
    print(f"Architecture: CNN (3 -> 64 channels, 128 -> 64 resolution) + ViT (64 patches, {embed_dim}-dim)")
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_schedule, gamma=0.1)

    best_val = float("inf")
    best_state = None
    patience = 1e6
    bad_epochs = 0

    t0 = time.time()

    import tqdm
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for xb, yb in tqdm.tqdm(train_loader):
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = model.criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)

        scheduler.step()
        train_loss = running / len(ds_train)
        model.train_curve[epoch - 1] = train_loss

        # Validation
        model.eval()
        with torch.no_grad():
            vtotal = 0.0
            for xb, yb in val_loader:
                preds = model(xb)
                vloss = model.criterion(preds, yb)
                vtotal += vloss.item() * xb.size(0)
            val_loss = vtotal / len(ds_val)
            model.val_curve[epoch - 1] = val_loss

        # Early stopping
        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        # Progress
        now = time.time()
        time_per_epoch = (now - t0) / epoch
        secs_left = time_per_epoch * (epochs - epoch)
        etad = datetime.datetime.fromtimestamp(now + secs_left)
        eta = etad.strftime("%H:%M:%S")
        nowdate = datetime.datetime.fromtimestamp(now)
        lr_current = optimizer.param_groups[0]['lr']

        def format_time(seconds):
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        elps = format_time(now - t0)
        rem = format_time(secs_left)

        print(f"[{epoch:3d}/{epochs}] net{idd:d}  train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {lr_current:.2e} | bad {bad_epochs:02d} | ETA {eta} | Remain {rem} | Sofar {elps}")
        
        if nowdate.day - etad.day != 0:
            print('tomorrow')

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}. Best val {best_val:.4f}.")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model with val loss {best_val:.4f}")
    
    return model


def plot_loss_curve(model):
    import matplotlib.pyplot as plt
    import os
    
    epochs_completed = (model.train_curve != 0).sum().item()
    train_curve = model.train_curve[:int(epochs_completed)].cpu().numpy()
    val_curve = model.val_curve[:int(epochs_completed)].cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_curve, label='Train Loss', linewidth=2)
    ax.plot(val_curve, label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training Curves - net{idd} (Hybrid CNN-ViT)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    oname = f"{os.environ['HOME']}/plots/loss_curve_net{idd}.png"
    fig.savefig(oname, dpi=150, bbox_inches='tight')
    print(f"Loss curve saved: {oname}")
    plt.close(fig)
