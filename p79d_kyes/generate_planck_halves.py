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
import torch
import h5py
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter

def downsample_avg(x, M):
    if x.ndim == 2:   # [N, N]
        x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, N, N]
        out = F.adaptive_avg_pool2d(x, (M, M))
        return out.squeeze(0).squeeze(0) # -> [M, M]
    elif x.ndim == 4: # [B, C, N, N]
        return F.adaptive_avg_pool2d(x, (M, M))
    else:
        raise ValueError("Input must be [N, N] or [B, C, N, N]")
                     
mapfile = "data/HFI_SkyMap_353_2048_R2.02_full.fits"
import healpy as hp
import numpy as np
import planck_reader 

m,mE,mB = planck_reader.read_353()
plt.clf()
hp.mollview(m)
plt.savefig('%s/plots/m'%os.environ['HOME'])
plt.clf()
hp.mollview(mE)
plt.savefig('%s/plots/mE'%os.environ['HOME'])
plt.clf()
hp.mollview(mB)
plt.savefig('%s/plots/mB'%os.environ['HOME'])

# Taurus Galactic center
Nplots=5000
size=400
reso=5.0
target = 128
smooth = 5
makeplots=True
suffix='allnorm_%d_sets_%d_smooth_%d_target'%(Nplots,smooth,target)
for half in [0,1]:
    TEB1 = np.zeros([Nplots, 3, size,size])
    nactual=0
    while nactual < Nplots:
        BBB = np.random.uniform(-90,90)
        if half==0:
            LLL = np.random.uniform(0,180)
        elif half==1:
            LLL = np.random.uniform(180,360)
        plt.close('all')
        print('half %d nplot %d L,B (%0.2e, %0.2e)'%(half,nactual, LLL,BBB))
        def get_cutout(hmap, rot, xsize=400, reso=5.0):
            """
            Returns a cleaned, finite NumPy array cutout (no masked pixels).
            """
            cut = hp.gnomview(
                hmap,
                rot=rot,
                xsize=xsize,
                reso=reso,
                return_projected_map=True,
                no_plot=True
            )

            # Convert masked array → normal array, fill with NaN first
            cut = np.ma.filled(cut, np.nan)

            # Replace NaNs or infinities with 0 for ML use
            cut[~np.isfinite(cut)] = 0.0

            cut = np.asarray(cut, dtype=np.float32)


            return cut


# --- Extract all three maps
        cutout  = get_cutout(m,  (LLL,BBB), xsize=size, reso=reso)
        cutoutE = get_cutout(mE, (LLL,BBB), xsize=size, reso=reso)
        cutoutB = get_cutout(mB, (LLL,BBB), xsize=size, reso=reso)

        if len(np.where(cutout==0)[0])/cutout.size > 0.3:
            print("too much mask, skip")
            continue
        #pdb.set_trace()


        cutout = gaussian_filter(cutout,smooth)
        cutoutE = gaussian_filter(cutoutE,smooth)
        cutoutB = gaussian_filter(cutoutB,smooth)

        TEB1[nactual,0]=cutout
        TEB1[nactual,1]=cutoutE
        TEB1[nactual,2]=cutoutB
        nactual+=1
            
        def maxer(arr):
            #mmm = np.abs(arr).max()
            #return [-mmm,mmm]
            return arr.min(),arr.max()
        if makeplots:
            fig,ax=plt.subplots(1,3,figsize=(12,4))
            im = ax[0].imshow(cutout, origin="lower",
                            norm=Normalize(vmin=maxer(cutout)[0], vmax=maxer(cutout)[1]),
                            cmap="inferno")  # pick your favorite colormap
            fig.colorbar(im,ax=ax[0])
            im = ax[1].imshow(cutoutE, origin="lower",
                            norm=Normalize(vmin=maxer(cutoutE)[0], vmax=maxer(cutoutE)[1]),
                            cmap="inferno")  # pick your favorite colormap
            fig.colorbar(im,ax=ax[1])
            im = ax[2].imshow(cutoutB, origin="lower",
                            norm=Normalize(vmin=maxer(cutoutB)[0], vmax=maxer(cutoutB)[1]),
                            cmap="inferno")  # pick your favorite colormap
            fig.colorbar(im,ax=ax[2])
            fig.savefig('%s/plots/subset_%s%04d'%(os.environ['HOME'],suffix, Nplots))
            plt.close(fig)
                             
    oname = "planck_%s_half%d.h5"%(suffix,half)
    fptr = h5py.File(oname,'w')
    TEB1 = torch.tensor(TEB1)

    fptr['subsets']=downsample_avg(TEB1,target)
    fptr.close()
    print(oname)
