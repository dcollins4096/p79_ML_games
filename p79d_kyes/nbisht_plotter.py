import torch.nn.functional as F
from importlib import reload
import importlib
import sys
import os
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import time
import loader
import matplotlib as mpl
import dtools_global.vis.pcolormesh_helper as pch
from scipy.stats import pearsonr
import tqdm
import torch_power
from collections import defaultdict
#import hotness
from torch.utils.data import Dataset, DataLoader

def plot_scalar_onlyms(net_name,train_loader, val_loader, tst_loader, model):

    subs = ['test']
    device = 'cuda'
    ms_net=[]
    ms_target=[]
    for ns,subset in enumerate(subs):
        mmin=20
        mmax=-20
        this_set = {'train':train_loader, 'valid':val_loader, 'test':tst_loader}[subset]
        fig,ax=plt.subplots(1,1)
        
        maxs=0
        with torch.no_grad():
            for xb, yb in tqdm.tqdm(this_set):
                ms = yb[0][0].cpu()
                moo = model( xb )
                ms_moo = moo[-1][0].cpu()
                ms_net.append(ms_moo.item())
                ms_target.append(ms)
                maxs=max([maxs,ms])
        if len(ms_target) < 100:
            for a,b in zip(ms_target,ms_net):
                ax.scatter(a,b)
        else:
            pch.simple_phase(ms_target,ms_net,ax=ax)
        pearson_r = pearsonr(ms_target, ms_net)[0]
        ax.set_title(f'{subset.capitalize()} Set - Pearson R = {pearson_r:.4f}', fontsize=14)
        ax.plot( [0,maxs],[0,maxs])
        fig.savefig('%s/plots/%s_scalars_%s'%(os.environ['HOME'],net_name,subset))

    ms_target_np = np.array(ms_target)
    ms_net_np = np.array(ms_net)

    # Bin by Mach number and compute error
    bins = [0, 3, 6, 9, 12, 20]
    for i in range(len(bins)-1):
        mask = (ms_target_np >= bins[i]) & (ms_target_np < bins[i+1])
        if mask.sum() > 0:
            bin_pearson = pearsonr(ms_target_np[mask], ms_net_np[mask])[0]
            bin_mae = np.abs(ms_target_np[mask] - ms_net_np[mask]).mean()
            bin_count = mask.sum()
            print(f"Ms [{bins[i]}-{bins[i+1]}): N={bin_count:4d}, R={bin_pearson:.4f}, MAE={bin_mae:.4f}")

    
    