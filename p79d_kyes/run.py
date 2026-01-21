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
import torch_power
reload(loader)
reload(torch_power)

new_model   = 1
load_model  = 0
train_model = 1
save_model = 1
plot_models = 1

def nparam(model):
    return sum( param.numel() for param in model.parameters() if param.requires_grad)

if new_model:
    import networks.net4004  as net
    reload(net)
    all_data = net.load_data()
    model = net.thisnet()
    model.idd = net.idd


if load_model:
    model.load_state_dict(torch.load("models/test%d.pth"%net.idd))

if train_model:

    t0 = time.time()

    #import networks.net0064 as othernet
    print("Train model ",model.idd)
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

import plotter
reload(plotter)
reload(plotter)
if plot_models:
    net_name = "net%04d"%net.idd
    model.eval()
    plotter.plot1(net_name, model, all_data, 'WTF', subset='test')
    #plotter.plot1(net_name, model, all_data, 'WTF', subset='train')
    #plotter.plot1(net_name, model, all_data, 'WTF', subset='valid')
    #plotter.plot_hot(net_name, model, all_data, 'WTF')

if False:
    if hasattr(model,'train_curve'):
        net.plot_loss_curve(model)
    else:
        print('didnt save train curve')
    if hasattr(net,'DatasetNorm'):
        ds_train = net.DatasetNorm(all_data['train'], compute_stats=True)
        ds_val   = net.DatasetNorm(all_data['valid'], mean_x=ds_train.mean_x, std_x=ds_train.std_x, mean_y=ds_train.mean_y, std_y=ds_train.std_y)
        #ds_tst   = net.DatasetNorm(all_data['test'], mean_x=ds_train.mean_x, std_x=ds_train.std_x, mean_y=ds_train.mean_y, std_y=ds_train.std_y)

    else:
        ds_train = net.SphericalDataset(all_data['train'])
        ds_val   = net.SphericalDataset(all_data['valid'])
        ds_tst   = net.SphericalDataset(all_data['test'])
    print('ploot')
    delta = []
    err={'train':[],'valid':[],'test':[]}
    detail_err = {'train':[],'valid':[],'test':[]}
    subs = ['train','valid','test']
    subs = ['train','valid']
    subs = ['train']
    fig1,ax=plt.subplots(2,3,figsize=(12,4))

    for ns,subset in enumerate(subs):
        mmin=20
        mmax=-20
        #this_set = {'train':ds_train, 'valid':ds_val, 'test':ds_tst}[subset]
        this_set = ds_train #{'train':ds_train, 'valid':ds_val}[subset]

        with torch.no_grad():
            for n in tqdm.tqdm(range(len(this_set))):
                if n <-1:
                    continue

                if 1:
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

                    detail_err[subset].append(  model.criterion1(moo,EB.unsqueeze(0)))
                    err[subset].append( model.criterion(moo,EB.unsqueeze(0)))
            errs = np.array([e.item() for e in err[subset]])
            args = np.argsort(errs)
            if subset == 'test':
                dothese = np.concatenate([np.arange(10),args[:10],args[-10:]])
            else:
                dothese = np.arange( len(this_set))
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

                thiserr = errs[n]
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
                    fig.savefig('%s/plots/show_net%d_%s_%04d'%(os.environ['HOME'],model.idd,subset,i))
                    plt.close(fig)



        #this_err = torch.tensor(err[subset]).detach().numpy()
        ax[0][ns].hist(errs)
        thiserr = torch.stack(detail_err[subset]).cpu().detach().numpy()
        args = np.argsort( thiserr, axis=0)
        labels = ['1','2','3','4','SSIM','Grad','Pear','Power']
        col = ['c','m','y','r','g','b','brown','purple']
        for nerr in range( thiserr.shape[1]):
            a = thiserr[args,nerr].flatten()
            a.sort()
            ax[1][ns].hist( a, label=labels[nerr], histtype='step', color=col[nerr])
        ax[1][ns].legend(loc=0)



    
    fig1.tight_layout()
    oname = '%s/plots/errhist_net%d'%(os.environ['HOME'],model.idd)
    print(oname)
    fig1.savefig(oname)



