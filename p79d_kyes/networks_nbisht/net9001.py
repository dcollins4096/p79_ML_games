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

idd = 9001
what = "Vision Transformer for Mach number prediction"

dirpath = "/home/dcollins/repos/p79_ML_games/p79d_kyes/datasets/"

fname_train = "p79d_subsets_S128_N1_xyz_suite7vs_first.h5"
fname_valid = "p79d_subsets_S128_N1_xyz_suite7vs_second.h5"

ntrain = 14000
nvalid = 30
ntest = 5000
downsample = None  # ViT works with fixed patch sizes
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 30
lr = 5e-4
batch_size = 64
lr_schedule = [1000]
weight_decay = 1e-2

# ViT hyperparameters
img_size = 128
patch_size = 16  # 128/16 = 8x8 patches Is actually patch_num
embed_dim = 384
depth = 6  # number of transformer blocks
num_heads = 6
mlp_ratio = 4.0
dropout = 0.1


def load_data():
    print('read the data')
    train = loader.loader(dirpath + fname_train, ntrain=ntrain, nvalid=nvalid)
    valid = loader.loader(dirpath + fname_valid, ntrain=1, nvalid=nvalid)
    all_data = {
        'train': train['train'],
        'valid': valid['valid'],
        'test': valid['test'][:ntest],
        'quantities': {}
    }
    all_data['quantities']['train'] = train['quantities']['train']
    all_data['quantities']['valid'] = valid['quantities']['valid']
    all_data['quantities']['test'] = valid['quantities']['test']
    print('done')
    return all_data


def thisnet():
    model = VisionTransformer(
        img_size=img_size,
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
    def __init__(self, all_data, quan, rotation_prob=0.0, rand=False):
        self.rotation_prob = rotation_prob
        self.quan = quan
        self.all_data = all_data
        self.rand = rand

    def __len__(self):
        return self.all_data.size(0)

    def __getitem__(self, idx):
        H, W = self.all_data[0][0].shape
        if self.rand:
            dy = torch.randint(0, H, (1,)).item()
            dx = torch.randint(0, W, (1,)).item()
            theset = torch.roll(self.all_data[idx], shifts=(dy, dx), dims=(-2, -1))
        else:
            theset = self.all_data[idx]
        ms = self.quan['Ms_act'][idx]
        return theset[0:3].to(device), torch.tensor([ms], dtype=torch.float32).to(device)


def set_seed(seed=8675309):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, n_patches_h, n_patches_w]
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
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
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP with GELU activation."""
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
    """Transformer block with attention and MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for regression."""
    def __init__(self, img_size=128, patch_size=16, in_chans=3, num_classes=1,
                 embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches

        # Learnable position embeddings + class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Regression head
        self.head = nn.Linear(embed_dim, num_classes)

        # Training curves
        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))

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

    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, n_patches+1, embed_dim]

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Use class token for prediction
        x = x[:, 0]  # [B, embed_dim]
        x = self.head(x)  # [B, num_classes]

        return x

    def criterion1(self, preds, target):
        return {'mse': F.mse_loss(preds, target)}

    def criterion(self, preds, target):
        losses = self.criterion1(preds, target)
        return sum(losses.values())


def trainer(model, all_data, epochs=200, batch_size=32, lr=1e-4,
            weight_decay=1e-4, lr_schedule=[900]):
    set_seed()

    ds_train = SphericalDataset(all_data['train'], all_data['quantities']['train'], rand=True)
    ds_val = SphericalDataset(all_data['valid'], all_data['quantities']['valid'])
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(ds_val, batch_size=max(64, batch_size), shuffle=False, drop_last=False)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_schedule, gamma=0.1)

    best_val = float("inf")
    best_state = None
    patience = 1e6
    bad_epochs = 0

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

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
                xb = xb.to(device)
                yb = yb.to(device)
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

        # Progress reporting
        now = time.time()
        time_per_epoch = (now - t0) / epoch
        secs_left = time_per_epoch * (epochs - epoch)
        etad = datetime.datetime.fromtimestamp(now + secs_left)
        eta = etad.strftime("%H:%M:%S")
        lr_current = optimizer.param_groups[0]['lr']

        def format_time(seconds):
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        elps = format_time(now - t0)
        rem = format_time(secs_left)

        print(f"[{epoch:3d}/{epochs}] net{idd:d}  train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {lr_current:.2e} | bad {bad_epochs:02d} | ETA {eta} | Remain {rem} | Sofar {elps}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}. Best val {best_val:.4f}.")
            break

    return model


def plot_loss_curve(model):
    """Plot training and validation loss curves."""
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
    ax.set_title(f'Training Curves - net{idd}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    oname = f"{os.environ['HOME']}/plots/loss_curve_net{idd}.png"
    fig.savefig(oname, dpi=150, bbox_inches='tight')
    print(f"Loss curve saved: {oname}")
    plt.close(fig)