
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
import hotness

def plot_three_flow(Eguess,Etarget,E_samples,fig=None,axs=None, title='', floating=False):
    if hasattr(Eguess, 'cpu'):
        Eguess=Eguess.cpu()
    if hasattr(Etarget,'cpu'):
        Etarget=Etarget.cpu()
    Emin = min([Etarget.min(), Eguess.min()])
    Emax = max([Etarget.max(), Eguess.max()])
    #Enorm = mpl.colors.Normalize(vmin=-1,vmax=1)
    Enorm = mpl.colors.Normalize(vmin=Emin,vmax=Emax)
    #Enorm = None

    ppp=axs[0].imshow(Etarget,norm=Enorm)
    fig.colorbar(ppp,ax=axs[0])
    axs[0].set(title='%s actual'%title, xlabel='x [pixel]', ylabel = 'y [pixel]')

    ppp=axs[1].imshow(Eguess,norm=Enorm)
    fig.colorbar(ppp,ax=axs[1])
    L2 = (((Etarget-Eguess)**2).mean())**0.5
    axs[1].set(title='%s predict %0.2e'%(title,L2), xlabel='x [pixel]', ylabel = 'y [pixel]')

    mean = E_samples.mean(axis=0)
    std  = E_samples.std(axis=0)
    #mean = E_samples[0,...]
    #std  = E_samples[1,...]
    ppp=axs[2].imshow(mean.cpu().detach().numpy(), norm=Enorm)
    L2 = (((Etarget-mean.cpu())**2).mean())**0.5
    axs[2].set(title='%s NF mean %0.2e'%(title,L2), xlabel='x [pixel]', ylabel = 'y [pixel]')
    fig.colorbar(ppp,ax=axs[2])
    ppp=axs[3].imshow(std.cpu().detach().numpy(), norm=None)
    fig.colorbar(ppp,ax=axs[3])

    E1 = Etarget.flatten()
    E2 = Eguess.flatten()
    pch.simple_phase(E1,E2,ax=axs[4], colorbar=False)
    axs[4].plot( [Emin,Emax],[Emin,Emax],c='k')


def plot_three_err(Eguess,Etarget,var,out_prob,mean,model,fig=None,axs=None, title='', floating=False):
    if hasattr(Eguess, 'cpu'):
        Eguess=Eguess.cpu()
    if hasattr(Etarget,'cpu'):
        Etarget=Etarget.cpu()
    Emin = min([Etarget.min(), Eguess.min()])
    Emax = max([Etarget.max(), Eguess.max()])
    if not floating:
        #Enorm = mpl.colors.Normalize(vmin=Emin,vmax=Emax)
        Enorm = mpl.colors.SymLogNorm(1.0,vmin=Emin,vmax=Emax)
    else:
        raise
        Enorm = None
    Enorm = mpl.colors.Normalize(vmin=-1,vmax=1)
    ppp=axs[0].imshow(Etarget,norm=Enorm)
    fig.colorbar(ppp,ax=axs[0])
    axs[0].set(title='%s actual'%title, xlabel='x [pixel]', ylabel = 'y [pixel]')
    ppp=axs[1].imshow(Eguess,norm=Enorm)
    fig.colorbar(ppp,ax=axs[1])
    axs[1].set(title='%s predict'%title, xlabel='x [pixel]', ylabel = 'y [pixel]')
    ppp=axs[2].imshow(mean.cpu().detach().numpy(), norm=Enorm)
    fig.colorbar(ppp,ax=axs[2])
    ppp=axs[3].imshow(var.cpu().detach().numpy())
    fig.colorbar(ppp,ax=axs[3])
    hotness.plot_hot( out_prob, model.range.cpu().detach().numpy(), model.num_bins.cpu().detach().numpy(), axs[4])


    E1 = Etarget.flatten()
    E2 = Eguess.flatten()
    pch.simple_phase(E1,E2,ax=axs[5], colorbar=False)
    axs[5].plot( [Emin,Emax],[Emin,Emax],c='k')



    


def plot_three(Eguess,Etarget,fig=None,axs=None, title='', floating=False):
    print(Eguess.shape)
    if hasattr(Eguess, 'cpu'):
        Eguess=Eguess.cpu()
    if hasattr(Etarget,'cpu'):
        Etarget=Etarget.cpu()
    Emin = min([Etarget.min(), Eguess.min()])
    Emax = max([Etarget.max(), Eguess.max()])
    if not floating:
        #Enorm = mpl.colors.Normalize(vmin=Emin,vmax=Emax)
        Enorm = mpl.colors.SymLogNorm(1.0,vmin=Emin,vmax=Emax)
    else:
        raise
        Enorm = None
    ppp=axs[0].imshow(Etarget,norm=Enorm)
    #fig.colorbar(ppp,ax=axs[0])
    axs[0].set(title='%s actual'%title, xlabel='x [pixel]', ylabel = 'y [pixel]')
    ppp=axs[1].imshow(Eguess,norm=Enorm)
    #fig.colorbar(ppp,ax=axs[1])
    axs[1].set(title='%s predict'%title, xlabel='x [pixel]', ylabel = 'y [pixel]')
    er = pearsonr( Eguess.flatten(), Etarget.flatten())[0]
    #fig.colorbar(ppp,ax=axs[2])
    axs[2].set(title='pearson %0.4f'%er)
    E1 = Etarget.flatten()
    E2 = Eguess.flatten()
    pch.simple_phase(E1,E2,ax=axs[2], colorbar=False)
    axs[2].plot( [Emin,Emax],[Emin,Emax],c='k')
    axs[2].set(xlabel='Actual pixel value',ylabel='Predicted')
    import dtools_global.math.power_spectrum as ps
    if len(axs)>=4:
        #power_guess = torch_power.powerspectrum(Eguess)
        #power_target = torch_power.powerspectrum(Etarget)
        power_guess = ps.powerspectrum(Eguess.detach().numpy())
        power_target = ps.powerspectrum(Etarget.detach().numpy())
        axs[3].plot( power_guess.kcen, power_guess.avgpower, c='r')
        axs[3].plot( power_target.kcen, power_target.avgpower, c='k')
        axs[3].set(xscale='log',yscale='log', title='Power spectrum %s'%title, xlabel='k', ylabel='power')
    if len(axs)==5:
        cross=ps.cross_spectrum(Eguess.detach().numpy(), Etarget.detach().numpy())
        axs[4].plot(power_target.kcen, power_target.avgpower, c='k')
        axs[4].plot(cross.kcen, cross.avgpower, c='r')
        axs[4].set(xscale='log',yscale='log', title='Cross spectrum %s'%title, xlabel='k', ylabel='power')


def plot_loss_curve(net_name,model, suffix):
    plt.clf()
    plt.plot(model.train_curve.cpu(), label="train")
    plt.plot(model.val_curve.cpu(),   label="val") 
    plt.yscale("log")
    plt.ylim([1e-2,10])


    plt.legend() 
    plt.tight_layout()
    plt.savefig("%s/plots/%s_%s_err_time.png"%(os.environ['HOME'], net_name, suffix))



def plot1(net_name,model, all_data, suffix, erronly=True):
    do_bispectrum=False
    net = importlib.import_module(f"networks.{net_name}")

    plot_loss_curve(net_name, model, suffix)

    ds_train = net.SphericalDataset(all_data['train'].to('cpu'))
    #ds_val   = net.SphericalDataset(all_data['valid'].to('cpu'))
    #ds_tst   = net.SphericalDataset(all_data['test'].to('cpu'))
    ds_val = None
    ds_tst = None
    device = 'cuda'
    model = model.to(device)

    #this_set = ds_val
    subset = 'train'
    if subset == 'test':
        this_set = ds_tst
        subs=['test']
    elif subset == 'train':
        this_set = ds_train
        subs=['train']
    elif subset == 'valid':
        this_set = ds_val
        subs = ['valid']

    Nsamples=10
    print('Subset',subset, 'nsamples',Nsamples)
    error_list = defaultdict(list)
    with torch.no_grad():
        for n in tqdm.tqdm(range(len(this_set))):
            if n >50000:
                if 'kludge' not in dir():
                    kludge = True
                    print("KLUDGE for speed")
                continue
            Tmode = this_set[n][0]
            if len(Tmode.shape) == 3:
                Tmode = Tmode.squeeze(0)
            moo = model( Tmode.unsqueeze(0).to(device), return_features=True)

            if type(moo) == tuple:
                #moo = [mmm.cpu() for mmm in moo]
                nchannels = moo[0].shape[1]
                if nchannels == 2 or nchannels == 3:
                    EB = this_set[n][1]
                else:
                    EB = this_set[n][1][0:1]
            else:
                #moo = moo.cpu()
                if moo[0].shape[1] == 2:
                    EB = this_set[n]
                else:
                    EB = this_set[n][1]

            err_dict= model.criterion1(moo,EB.unsqueeze(0).to(device))
            for err in err_dict:
                error_list[err].append(err_dict[err].item())
            error_list['total'].append( sum(err_dict.values()).item())
    #if erronly:
    #    return error_list
    fig,ax=plt.subplots(1,1)
    colors = {'L1_0':'c','L1_1':'m','L1_2':'y','L1_3':'m','L1_4':'m','L1_Multi':'m','Pear':'r','Grad':'g','SSIM':'b','Power':'purple', 'total':'k'}
    for err in error_list:
        eee =  error_list[err]
        x = sorted(eee)
        y = np.arange(len(x))/len(x)
        ax.plot(x,y,label=err,color=colors.get(err,'orange'))
    ax.legend(loc=0)
    ax.set(xscale='log',xlim=[1e-2,5])
    fig.savefig('%s/plots/%s_%s_%s_errors.png'%(os.environ['HOME'],net_name,suffix,subset))


    #subs = ['train','valid','test']
    #subs = ['test','valid', 'train']
    #subs = ['valid']
    for ns,subset in enumerate(subs):
        mmin=20
        mmax=-20
        this_set = {'train':ds_train, 'valid':ds_val, 'test':ds_tst}[subset]
        #args = np.argsort(error_list['Power'])
        #N = len(args)
        #dothese = {'valid':[12,20], 'test':[0,1], 'train':range(10)}[subset]
        #args[0],args[int(0.25*N)], args[int(0.5*N)], args[int(0.75*N)],args[-1]
        #    #dothese = np.where( np.array(error_list['Power'] ) > 2)[0]
        if subset == 'test':
            ppp = np.argsort(error_list['total'])
            dothese = list( ppp[::ppp.size//30])
            dothese = ppp[:3]
        elif subset == 'valid':
            dothese = range(len(this_set))
        elif subset == 'train':
            dothese = range( min([30, len(this_set)]))

            
            #dothese = ppp[::50]
            #dothese = [ 939, 627, 448,1107, 543] +list( ppp[::ppp.size//30])
        #dothese=range(len(this_set))
        print("DO", dothese)

        with torch.no_grad():
            for i,n in tqdm.tqdm(enumerate(dothese)):
                Tmode = this_set[n][0]
                if len(Tmode.shape) == 3:
                    Tmode = Tmode.squeeze(0)
                moo = model( Tmode.unsqueeze(0).to(device), return_features=True)
                EB_samples=None
                if hasattr(model, 'flow_head'):
                    #res_samples = model.flow_head.sample_n(moo[1], n_samples=100)   # [K,B,2,H,W]
                    #EB_samples = res_samples + model(Tmode.unsqueeze(0).to(net.device), return_features=False)[1]  # add back predicted EB
                    EB_samples = model.flow_head.sample_n(moo[1], n_samples=Nsamples)
                if type(moo) == tuple:
                    EBguess = moo[0].cpu().squeeze(0)
                    EB = this_set[n][1]
                else:
                    EBguess = moo.squeeze(0)
                    EB = this_set[n][1]

                if EB_samples is not None:
                    fig,axes=plt.subplots(3,5,figsize=(15,8))
                    ax0,ax1,ax2=axes
                    fig.tight_layout()
                    fig.savefig('%s/plots/%s_%s_%s_%04d.png'%(os.environ['HOME'],net_name,suffix,subset,i))
                    plt.close(fig)
                    plot_three_flow(EBguess[0,:,:], EB[0,:,:], EB_samples[:,0,0,:,:],title='T', axs=ax0, fig=fig)
                    plot_three_flow(EBguess[1,:,:], EB[1,:,:], EB_samples[:,0,1,:,:],title='E', axs=ax1, fig=fig)
                    plot_three_flow(EBguess[2,:,:], EB[2,:,:], EB_samples[:,0,2,:,:],title='B', axs=ax2, fig=fig)
                    fig.tight_layout()
                    fig.savefig('%s/plots/%s_%s_%s_%04d.png'%(os.environ['HOME'],net_name,suffix,subset,i))
                    plt.close(fig)
                else:
                    fig,axes=plt.subplots(3,5,figsize=(15,8))
                    ax0,ax1,ax2=axes
                    EBguess = torch.nan_to_num(EBguess,nan=0.0)
                    EB = torch.nan_to_num(EB,nan=0.0)
                    if EBguess.shape[0]==3:
                        plot_three(EBguess[0,:,:], EB[0,:,:],title='T', axs=ax0, fig=fig)
                        plot_three(EBguess[1,:,:], EB[1,:,:],title='E', axs=ax1, fig=fig)
                        plot_three(EBguess[2,:,:], EB[2,:,:],title='B', axs=ax2, fig=fig)
                    elif EBguess.shape[0]==2:
                        #plot_three(EBguess[0,:,:], EB[0,:,:],title='T', axs=ax0)
                        for aaa in ax0:
                            aaa.imshow(Tmode.detach().cpu().numpy())
                        plot_three(EBguess[0,:,:], EB[0,:,:],title='E', axs=ax1, fig=fig)
                        plot_three(EBguess[1,:,:], EB[1,:,:],title='B', axs=ax2, fig=fig)

                    fig.tight_layout()
                    fig.savefig('%s/plots/%s_%s_%s_%04d.png'%(os.environ['HOME'],net_name,suffix,subset,i))
                    plt.close(fig)

    return model

