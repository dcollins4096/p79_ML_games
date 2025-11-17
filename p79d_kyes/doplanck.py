import healpy as hp  
import matplotlib.pyplot as plt
import numpy as np   
import os 
from matplotlib.colors import LogNorm, Normalize,SymLogNorm
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
import dtools_global.vis.pcolormesh_helper as pch
from scipy.stats import pearsonr
import tqdm
import torch_power
import plotter
import planck_reader
from scipy.ndimage import gaussian_filter
                     
if 'm' not in dir():
    m, mE, mB = planck_reader.read_353()

if 1:

# Taurus Galactic center
    l_taurus, b_taurus = 170.0, -15.0
#    l_taurus, b_taurus = 159, -74 #actuall L1457
    l_taurus, b_taurus = 140, 50 #actually Polaris Flare
    size = 400
# Extract cutout as an array (not just a plot)
    cutout = hp.gnomview(m, rot=(l_taurus, b_taurus),
                         xsize=size, reso=5.0,
                         return_projected_map=True,
                         no_plot=True)   # suppress auto-plot
    cutoutE = hp.gnomview(mE, rot=(l_taurus, b_taurus),
                         xsize=size, reso=5.0,
                         return_projected_map=True,
                         no_plot=True)   # suppress auto-plot
    cutoutB = hp.gnomview(mB, rot=(l_taurus, b_taurus),
                         xsize=size, reso=5.0,
                         return_projected_map=True,
                         no_plot=True)   # suppress auto-plot
    print("L,B",l_taurus,b_taurus)
    filt=10
    cutout = gaussian_filter(cutout,filt)
    cutoutE = gaussian_filter(cutoutE,filt)
    cutoutB = gaussian_filter(cutoutB,filt)
    cutout = torch.tensor(cutout, dtype=torch.float32)
    cutoutE = torch.tensor(cutoutE, dtype=torch.float32)
    cutoutB = torch.tensor(cutoutB, dtype=torch.float32)
        
# Mask negative values (log scale needs >0) 
#cutout = np.ma.masked_less_equal(cutout, 0)

# Plot manually with log scaling
if 1:
    print('SUM',torch.abs(cutout).sum())
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    im = ax[0].imshow(cutout, origin="lower",
                    norm=SymLogNorm(0.5,vmin=cutout.min(), vmax=cutout.max()),
                    cmap="inferno")  # pick your favorite colormap
    im = ax[1].imshow(cutoutE, origin="lower",
                    norm=Normalize(vmin=cutoutE.min(), vmax=cutoutE.max()),
                    cmap="inferno")  # pick your favorite colormap
    im = ax[2].imshow(cutoutB, origin="lower",
                    norm=SymLogNorm(1e-4,vmin=cutoutB.min(), vmax=cutoutB.max()),
                    cmap="inferno")  # pick your favorite colormap
    fig.savefig('%s/plots/test1'%os.environ['HOME'])
                     

new_model   = 1
load_model  = 1
train_model = 0

if 1:
    if new_model:
        import networks.net0181 as net
        reload(net)
        #all_data = net.load_data()
        model = net.thisnet()
        model.idd = net.idd


    if load_model:
        #model.load_state_dict(torch.load("models/test%d.pth"%net.idd))
        #model.load_state_dict(torch.load("models/test%d.pth"%net.idd))
        fname = "models/net0155_"
        model.load_state_dict(torch.load(fname))

    net_name='planck'
    suffix=''
    subset=''
    i=0

    with torch.no_grad():
        Tmode = cutout
        moo = model( Tmode.unsqueeze(0).to(net.device))
        do_b=False
        Tguess=moo[0].squeeze(0)[0].detach()
        Eguess=moo[0].squeeze(0)[1].detach()
        Bguess=moo[0].squeeze(0)[2].detach()
        Tsky = cutout
        Esky = cutoutE
        Bsky = cutoutB

        if 1:
            fig,axes=plt.subplots(3,5,figsize=(14,8))
            ax0,ax1,ax2=axes
            plotter.plot_three(Tguess, Tsky,title='T', axs=ax0, fig=fig)
            plotter.plot_three(Eguess, Esky,title='E', axs=ax1, fig=fig)
            plotter.plot_three(Bguess, Bsky,title='B', axs=ax2, fig=fig)

            fig.tight_layout()
            outname = '%s/plots/%s_%s_%s_%04d.png'%(os.environ['HOME'],net_name,suffix,subset,i)
            fig.savefig(outname)
            print(outname)
            plt.close(fig)



    if 0:
        with torch.no_grad():
            moo = model(sky.unsqueeze(0))
            fig, axes=plt.subplots(2,3,figsize=(12,6))
            T=moo[0].squeeze(0)[0].detach().cpu().numpy()
            E=moo[0].squeeze(0)[1].detach().cpu().numpy()
            B=moo[0].squeeze(0)[2].detach().cpu().numpy()

            Tmin = min([cutout.min(),T.min()])
            Emin = min([cutoutE.min()])
            Bmin = min([cutoutB.min()])
            Tmax = max([cutout.max(),T.max()])
            Emax = max([cutoutE.max()])
            Bmax = max([cutoutB.max()])
            ploot = axes[0][0].imshow( cutout, norm=LogNorm(vmin=Tmin, vmax=Tmax))
            fig.colorbar(ploot,ax=axes[0][0])
            ploot = axes[0][1].imshow( cutoutE,norm=SymLogNorm(1e-3,vmin=Emin, vmax=Emax))
            fig.colorbar(ploot,ax=axes[0][1])
            ploot = axes[0][2].imshow( cutoutB,norm=SymLogNorm(1e-3,vmin=Bmin, vmax=Bmax))
            fig.colorbar(ploot,ax=axes[0][2])

            ploot = axes[1][0].imshow(T, norm=LogNorm(vmin=T.min(), vmax=T.max()))
            fig.colorbar(ploot,ax=axes[1][0])
            ploot = axes[1][1].imshow(E, norm=SymLogNorm(1e-3,vmin=E.min(), vmax=E.max()))
            fig.colorbar(ploot,ax=axes[1][1])
            ploot = axes[1][2].imshow(B, norm=SymLogNorm(1e-3,vmin=B.min(), vmax=B.max()))
            fig.colorbar(ploot,ax=axes[1][2])
            fig.tight_layout()
            fig.savefig(f'{os.environ["HOME"]}/plots/magic')
            
