from importlib import reload
import sys
import os
sys.path.append('/home/dcollins/repos/')
import dtools_global.vis.pcolormesh_helper as pch
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import time
import loader
import matplotlib as mpl
from scipy.stats import pearsonr
import tqdm
import torch_power
reload(loader)
reload(torch_power)
import nbisht_plotter
reload(nbisht_plotter)
from torch.utils.data import Dataset, DataLoader

new_model   = 1
load_model  = 0
train_model = 1
save_model  = 1
plot_models = 1
net_name   = "net9005"

if new_model:
    import networks_nbisht.net9005 as net
    reload(net)
    all_data = net.load_data()
    model = net.thisnet()
    model.idd = net.idd

    print("Train Ms range:", all_data['quantities']['train']['Ms_act'].min(), 
                         all_data['quantities']['train']['Ms_act'].max())
    print("Valid Ms range:", all_data['quantities']['valid']['Ms_act'].min(), 
                            all_data['quantities']['valid']['Ms_act'].max())

    ms_train = all_data['quantities']['train']['Ms_act']
    ms_valid = all_data['quantities']['valid']['Ms_act']

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(ms_train, bins=50, alpha=0.7, label='Train')
    ax[0].hist(ms_valid, bins=50, alpha=0.7, label='Valid')
    ax[0].set(xlabel='Ms', ylabel='Count', title='Data Distribution')
    ax[0].legend()

    # Cumulative
    ax[1].hist(ms_train, bins=50, alpha=0.7, cumulative=True, label='Train')
    ax[1].hist(ms_valid, bins=50, alpha=0.7, cumulative=True, label='Valid')
    ax[1].set(xlabel='Ms', ylabel='Cumulative Count', title='Cumulative Distribution')
    ax[1].legend()

    fig.savefig(f"{os.environ['HOME']}/plots/ms_distribution_stratified_{net_name}.png")

    # Verify stratified validation
    ms_valid = all_data['quantities']['valid']['Ms_act']
    print("\nValidation set Mach distribution:")
    bins = [0, 4, 6, 8, 10, 15]
    for i in range(len(bins)-1):
        mask = (ms_valid >= bins[i]) & (ms_valid < bins[i+1])
        print(f"  Ms [{bins[i]:2d}-{bins[i+1]:2d}): {mask.sum():3d} samples")


if load_model:
    model.load_state_dict(torch.load("models/test%d.pth"%net.idd))
    model = model.to(net.device)
    model.eval()

if train_model:

    t0 = time.time()

    #import networks.net0064 as othernet
    net.train(model,all_data)
    if save_model:
        oname = "models/test%d.pth"%model.idd
        torch.save(model.state_dict(), oname)
        print("model saved ",oname)

    t1 = time.time() - t0
    hrs = t1//3600
    minute = (t1-hrs*3600)//60
    sec = (t1 - hrs*3600-minute*60)#//60
    total_time="%02d:%02d:%02d"%(hrs,minute,sec)


if plot_models:
    model.eval()
    net.plot_loss_curve(model)

    ds_train = net.SphericalDataset(all_data['train'], all_data['quantities']['train'])
    ds_val   = net.SphericalDataset(all_data['valid'], all_data['quantities']['valid'])
    ds_tst   = net.SphericalDataset(all_data['test'], all_data['quantities']['test'])

    train_loader = DataLoader(ds_train, batch_size=1, shuffle=False, drop_last=False)
    val_loader   = DataLoader(ds_val,   batch_size=1, shuffle=False, drop_last=False)
    tst_loader   = DataLoader(ds_tst,   batch_size=1, shuffle=False, drop_last=False)
    nbisht_plotter.plot_scalar_onlyms(net_name, train_loader, val_loader, tst_loader, model)