
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
from collections import defaultdict

import networks.net0141 as net
reload(net)
def plot(model, all_data, suffix):

    ds_train = net.SphericalDataset(all_data['train'])
    ds_val   = net.SphericalDataset(all_data['valid'])
    ds_tst   = net.SphericalDataset(all_data['test'])

    this_set = ds_tst

    error_list = defaultdict(list)
    with torch.no_grad():
        for n in tqdm.tqdm(range(len(this_set))):
            Tmode = this_set[n][0]
            if len(Tmode.shape) == 3:
                Tmode = Tmode.squeeze(0)
            moo = model( Tmode.unsqueeze(0).to(net.device))
            if type(moo) == tuple:
                moo = [mmm.cpu() for mmm in moo]
                nchannels = moo[0].shape[1]
                if nchannels == 2 or nchannels == 3:
                    EB = this_set[n][1]
                else:
                    EB = this_set[n][1][0:1]
            else:
                moo = moo.cpu()
                if moo[0].shape[1] == 2:
                    EB = this_set[n]
                else:
                    EB = this_set[n][1]

            err_dict= model.criterion1(moo,EB.unsqueeze(0))
            for err in err_dict:
                error_list[err].append(err_dict[err].item())
            error_list['total'].append( sum(err_dict.values()).item())
    fig,ax=plt.subplots(1,1)
    colors = {'L1_0':'c','L1_1':'m','L1_2':'y','L1_3':'m','L1_4':'m','L1_Multi':'m','Pear':'r','Grad':'g','SSIM':'b','Power':'purple', 'total':'k'}
    for err in error_list:
        eee =  error_list[err]
        x = sorted(eee)
        y = np.arange(len(x))/len(x)
        ax.plot(x,y,label=err,color=colors.get(err,'orange'))
    ax.legend(loc=0)
    ax.set(xscale='log',xlim=[1e-2,5])
    fig.savefig('%s/plots/errors_%s.png'%(os.environ['HOME'],suffix))


    subs = ['train','valid','test']
    for ns,subset in enumerate(subs):
        mmin=20
        mmax=-20
        this_set = {'train':ds_train, 'valid':ds_val, 'test':ds_tst}[subset]
        dothese = {'valid':[12,20], 'test':[np.argmax(error_list['total']), np.argmin(error_list['total'])], 'train':[0,1]}[subset]

        with torch.no_grad():
            for i,n in tqdm.tqdm(enumerate(dothese)):
                if i>30:
                    break
                Tmode = this_set[n][0]
                if len(Tmode.shape) == 3:
                    Tmode = Tmode.squeeze(0)
                moo = model( Tmode.unsqueeze(0).to(net.device))
                do_b=False
                if type(moo) == tuple:
                    EBguess = moo[0].cpu().squeeze(0)
                    EB = this_set[n][1]

                #plot_multipole.rmplot( sky[subset][n], rm, clm_model = moo, clm_real = clm, fname = "rm_and_sampled_%04d"%n)
                def plot_three(Eguess,Etarget,axs=None, title=''):
                    Emin = min([Etarget.min(), Eguess.min()])
                    Emax = max([Etarget.max(), Eguess.max()])
                    Enorm = mpl.colors.Normalize(vmin=Emin,vmax=Emax)
                    ppp=axs[0].imshow(Etarget,norm=Enorm)
                    fig.colorbar(ppp,ax=axs[0])
                    axs[0].set(title='%s actual'%title)
                    ppp=axs[1].imshow(Eguess,norm=Enorm)
                    fig.colorbar(ppp,ax=axs[1])
                    axs[1].set(title='%s guess'%title)
                    er = pearsonr( Eguess.flatten(), Etarget.flatten())[0]
                    fig.colorbar(ppp,ax=axs[2])
                    axs[2].set(title='pearson %0.4f'%er)
                    E1 = Etarget.flatten()
                    E2 = Eguess.flatten()
                    pch.simple_phase(E1,E2,ax=axs[2])
                    axs[2].plot( [Emin,Emax],[Emin,Emax],c='k')
                    import dtools_global.math.power_spectrum as ps
                    if len(axs)==4:
                        #power_guess = torch_power.powerspectrum(Eguess)
                        #power_target = torch_power.powerspectrum(Etarget)
                        power_guess = ps.powerspectrum(Eguess.detach().numpy())
                        power_target = ps.powerspectrum(Etarget.detach().numpy())
                        axs[3].plot( power_guess.kcen, power_guess.avgpower, c='r')
                        axs[3].plot( power_target.kcen, power_target.avgpower, c='k')
                        axs[3].set(xscale='log',yscale='log')



                if 1:
                    fig,axes=plt.subplots(3,4,figsize=(14,8))
                    ax0,ax1,ax2=axes
                    if EBguess.shape[0]==3:
                        plot_three(EBguess[0,:,:], EB[0,:,:],title='T', axs=ax0)
                        plot_three(EBguess[1,:,:], EB[1,:,:],title='E', axs=ax1)
                        plot_three(EBguess[2,:,:], EB[2,:,:],title='B', axs=ax2)
                    elif EBguess.shape[0]==2:
                        #plot_three(EBguess[0,:,:], EB[0,:,:],title='T', axs=ax0)
                        for aaa in ax0:
                            aaa.imshow(Tmode)
                        plot_three(EBguess[0,:,:], EB[0,:,:],title='E', axs=ax1)
                        plot_three(EBguess[1,:,:], EB[1,:,:],title='B', axs=ax2)

                    fig.tight_layout()
                    fig.savefig('%s/plots/image_%s_%s_%04d.png'%(os.environ['HOME'],suffix,subset,i))
                    plt.close(fig)



