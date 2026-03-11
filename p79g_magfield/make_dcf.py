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
import dtools.vis.pcolormesh_helper as pch
from scipy.stats import pearsonr
import tqdm

import dcf
reload(dcf)
import astropy.io.fits as pyfits

if 0:
    me_train = "p79d_subsets_S128_N1_xyz_suite7_tvsquh_half_first.h5"
    fname_valid = "p79d_subsets_S128_N1_xyz_suite7_tvsquh_half_second.h5"

    if 'all_data' not in dir():
        print('read the data')
        train= loader.loader(fname_train,ntrain=ntrain, nvalid=nvalid)
        valid= loader.loader(fname_valid,ntrain=1, nvalid=nvalid)
        all_data={'train':train['train'],'valid':valid['valid'], 'test':valid['test'], 'quantities':{}}
        all_data['quantities']['train']=train['quantities']['train']
        all_data['quantities']['valid']=valid['quantities']['valid']
        all_data['quantities']['test']=valid['quantities']['test']
        print('done')
        #return all_data

if 1:
    dataset=[]
    #drr = "datasets/p49_half_half_DD0011/DD0011_%s.fits"
    drr = "datasets/p49_4_2_DD0020/DD0020_%s.fits"
    axis='x'
    dataset.append(pyfits.open(drr%"density_%s"%axis)[0].data)
    dataset.append(pyfits.open(drr%"velocity_centroid_%s"%axis)[0].data)
    dataset.append(pyfits.open(drr%"velocity_variance_%s"%axis)[0].data)
    dataset.append(pyfits.open(drr%"Q%s"%axis)[0].data)
    dataset.append(pyfits.open(drr%"U%s"%axis)[0].data)
    dataset.append(pyfits.open(drr%"H_POS_%s"%axis)[0].data)

    dataset.append(pyfits.open(drr%"H_HORIZ_%s"%axis)[0].data)
    dataset.append(pyfits.open(drr%"H_VERT_%s"%axis)[0].data)

if 0:
    #this works, don't touch it.
    fig, axes = plt.subplots(3,2,figsize=(16,16))
    ax0,ax1,ax2=axes
    Q = dataset[3]
    U = dataset[4]
    Bx = dataset[6]
    By = dataset[7]
    theta_pol = 0.5*np.arctan2(U,Q)
    theta_mag = np.arctan2(By,Bx)
    norm = None#mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    p=ax0[0].imshow(theta_pol, cmap='hsv', norm=norm)
    ax0[0].set(title='theta pol')
    fig.colorbar(p,ax=ax0[0])
    p=ax0[1].imshow(theta_mag,cmap='hsv', norm=norm)
    ax0[1].set(title='theta field')
    fig.colorbar(p,ax=ax0[1])
    pch.simple_phase( theta_mag.flatten(), (theta_pol.flatten())%np.pi,ax=ax1[0])
    ax1[0].set(xlabel='theta field',ylabel='theta pol')
    #ax1[0].plot([0,np.pi],[0,np.pi])
    #ax1[0].plot([-np.pi,0],[0,np.pi])
    #ax1[0].plot([-np.pi/2,np.pi/2],[-np.pi/2,np.pi/2])
    ax1[0].plot([0,np.pi],[0,np.pi])
    ax1[0].plot([-np.pi,0],[0,np.pi])

    ppp=theta_pol.flatten()
    ax1[1].hist(np.abs(ppp) )
    ax1[1].set(title='P(theta pol)')
    ax2[0].imshow(Q)
    ax2[0].set(title='Q')
    ax2[1].imshow(U)
    ax2[1].set(title='U')
    fig.savefig('%s/plots/angles.png'%(os.environ['HOME']))

if 1:
    ds1, valid = dcf.do_it( dataset[0], dataset[2], dataset[3], dataset[4],dataset[5])
    fig, axes = plt.subplots(2,2, figsize=(16,16))
    ax0,ax1=axes
    ax0[0].imshow(dataset[5])
    ax0[0].set(title='B')
    ax0[1].imshow(ds1)
    ax0[1].set(title='DCF')
    pch.simple_phase( dataset[5][valid].flatten(), ds1[valid].flatten(),ax=ax1[0])
    ax1[0].set(xlabel='B true', ylabel='B dcf')
    fig.savefig('%s/plots/dcf.png'%(os.environ['HOME']))

if 0:
    #dataset = all_data['train'][0].detach().numpy()
    ds1, valid = dcf.do_it( dataset[0], dataset[2], dataset[3], dataset[4],dataset[5])
    fig, axes = plt.subplots(2,2, figsize=(16,16))
    ax0,ax1=axes
    Bx = dataset[-2]
    By = dataset[-1]
    ny, nx = Bx.shape

    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    ax0[0].imshow(dataset[5], origin='lower')
    Bt = dataset[5]
    #axes.quiver(X[::skip,::skip], Y[::skip,::skip], (By/Bt)[::skip,::skip], (Bx/Bt)[::skip,::skip])
    ax0[0].streamplot(X[::skip,::skip], Y[::skip,::skip], (Bx)[::skip,::skip], (By)[::skip,::skip])
    skip=16
    #axes[1].imshow( np.sqrt( Bx**2+By**2))
    if 0:
        ax0[1].quiver(X[::skip], Y[::skip], Bx[::skip], By[::skip])


        ax0[0].imshow(ds1)
        ax0[1].imshow(dataset[5])
        ax1[0].imshow(dataset[3])
        ax1[1].imshow(dataset[4])
    fig.savefig('%s/plots/dcf.png'%(os.environ['HOME']))
