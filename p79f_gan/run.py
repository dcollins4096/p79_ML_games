import torch.nn.functional as F
from importlib import reload
import sys
import os
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import time
import loader
import matplotlib as mpl
sys.path.append('/home/dcollins/repos/')
import dtools_global.vis.pcolormesh_helper as pch
from scipy.stats import pearsonr
import tqdm
import set_seed
reload(loader)

new_model   = 0
load_model  = 0
train_model = 1
save_model = 0
plot_models = 1

def nparam(model):
    return sum( param.numel() for param in model.parameters() if param.requires_grad)

if new_model:
    import networks.net0001  as net
    from   networks.net0001 import Generator, MachEmbed
    set_seed.set_seed()
    reload(net)
    all_data = net.load_data()
    #model = net.thisnet()
    #model.idd = net.idd


if load_model:
    model.load_state_dict(torch.load("models/test%d.pth"%net.idd))

if train_model:

    t0 = time.time()

    #import networks.net0064 as othernet
    print("Train model ",net.idd)
    G,G_ema,D,mach_embed=net.train(all_data, steps=3000)


    t1 = time.time() - t0
    hrs = t1//3600
    minute = (t1-hrs*3600)//60
    sec = (t1 - hrs*3600-minute*60)#//60
    total_time="%02d:%02d:%02d"%(hrs,minute,sec)

if plot_models:
    n=1
    z_dim=128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    z = torch.randn(n, z_dim, device=device)
    mach2=2.
    mach = torch.full((n,), mach2, device=device)
    c = mach_embed(mach)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    model = G
    x = model(z, c)  # [n, C, H, W]
    x = x.detach().float().cpu()
    x0 = x[:, 0, :, :]  # [n, H, W]
    img = x0[0].numpy()
    plt.imshow(img, origin="lower")
    plt.colorbar()
    plt.title(f"G (Mach={mach2}, )")
    model = G_ema
    plt.subplot(1,2,2)
    x = model(z, c)  # [n, C, H, W]
    x = x.detach().float().cpu()
    x0 = x[:, 0, :, :]  # [n, H, W]
    img = x0[0].numpy()
    plt.imshow(img, origin="lower")
    plt.colorbar()
    plt.title(f"G_ema (Mach={mach2}, )")
    plt.tight_layout()
    plt.savefig('%s/plots/gan1'%os.environ['HOME'], dpi=200)

