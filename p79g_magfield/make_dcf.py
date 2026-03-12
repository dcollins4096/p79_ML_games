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
    Bx = dataset[6]
    By = dataset[7]

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
    p=ax0[0].imshow(theta_pol, cmap='hsv', norm=norm, origin='lower')
    ax0[0].set(title='theta pol')
    fig.colorbar(p,ax=ax0[0])
    p=ax0[1].imshow(theta_mag,cmap='hsv', norm=norm, origin='lower')
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
    ax2[0].imshow(Q, origin='lower')
    ax2[0].set(title='Q')
    ax2[1].imshow(U, origin='lower')
    ax2[1].set(title='U')
    fig.savefig('%s/plots/angles.png'%(os.environ['HOME']))

if 1:
    #Bpos=dataset[5]
    Bx0 = dcf._local_mean(Bx, 9)
    By0 = dcf._local_mean(By, 9)
    theta_mag = np.arctan2(By,Bx)
    Bpos = np.sqrt(Bx0**2 + By0**2)
    result = dcf.do_it( dataset[0], dataset[2], dataset[3], dataset[4],dataset[5])

    Bdcf = result['B_dcf']
    valid = result['valid_mask']
    fig, axes = plt.subplots(4,3, figsize=(12,16))
    ax0,ax1,ax2, ax3=axes

    ax0[0].imshow(Bpos, origin='lower')
    ax0[0].set(title='B')
    ax0[1].imshow(Bdcf, origin='lower')
    ax0[1].set(title='DCF')
    pch.simple_phase( Bpos[valid].flatten(), Bdcf[valid].flatten(),ax=ax0[2])
    ax0[2].set(xlabel='B true',ylabel='B dcf')

    pch.simple_phase( theta_mag.flatten(), result['psi'].flatten(), ax=ax2[0])
    ax2[0].plot([-np.pi/2,np.pi/2],[-np.pi/2,np.pi/2], c='k')
    ax2[0].set(xlabel='Phi_B',ylabel='Phi_pol')
    def twochan(A,B, ax):
        # Normalize each image to [0,1]
        A = (A - A.min())/(A.max()-A.min())
        B = (B - B.min())/(B.max()-B.min())

        # Build RGB image
        rgb = np.zeros((A.shape[0], A.shape[1], 3))
        rgb[...,0] = A   # red channel
        rgb[...,1] = B   # green channel
        ax.imshow(rgb, origin='lower')

    p=ax1[0].imshow( result['sigma_phi']*180/np.pi, origin='lower')
    ax1[0].set(title=r'$\sigma_\phi$')
    #fig.colorbar(p,ax=ax1[1])
    ax1[1].imshow(np.cos(result['psi']), origin='lower', cmap='hsv')
    ax1[1].set(title=r'$\phi$')
    out = np.zeros_like(result['sigma_phi'])
    out[ result['sigma_phi'] > 0.5] = 1
    #out = result['sigma_phi']
    twochan( np.cos(result['psi']), out, ax1[2])
    ax1[2].set(title=r'$\phi$ red $\sigma_\phi$ green')

    twochan( Bpos, out, ax2[1])
    ax2[1].set(title=r'B real (red) $\sigma_\phi$ (green)')

    ax2[2].hist( result['sigma_phi'].flatten()*180/np.pi, bins=50)
    ax2[2].set(xlabel=r'$\sigma_\phi$', ylabel='N')

    ax3[0].imshow(result['sigma_v'], origin='lower')
    ax3[0].set(title=r'$\sigma_v$')

    twochan( result['sigma_v'], result['sigma_phi'], ax3[1])

    ax3[2].imshow(  result['sigma_v']/result['sigma_phi'])



    fig.tight_layout()
    fig.savefig('%s/plots/dcf.png'%(os.environ['HOME']))

