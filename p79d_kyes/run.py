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
reload(loader)

new_model   = 1
load_model  = 1
train_model = 0
save_model  = 1
plot_models = 1


if new_model:
    import networks.net0105 as net
    reload(net)
    all_data = net.load_data()
    model = net.thisnet()
    model.idd = net.idd


if load_model:
    model.load_state_dict(torch.load("models/test%d.pth"%net.idd))

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
    if hasattr(model,'train_curve'):
        net.plot_loss_curve(model)
    else:
        print('didnt save train curve')
    if hasattr(net,'DatasetNorm'):
        ds_train = net.DatasetNorm(all_data['train'], compute_stats=True)
        ds_val   = net.DatasetNorm(all_data['valid'], mean_x=ds_train.mean_x, std_x=ds_train.std_x, mean_y=ds_train.mean_y, std_y=ds_train.std_y)
        ds_tst   = net.DatasetNorm(all_data['test'], mean_x=ds_train.mean_x, std_x=ds_train.std_x, mean_y=ds_train.mean_y, std_y=ds_train.std_y)

    else:
        ds_train = net.SphericalDataset(all_data['train'])
        ds_val   = net.SphericalDataset(all_data['valid'])
        ds_tst   = net.SphericalDataset(all_data['test'])
    print('ploot')
    delta = []
    err={'train':[],'valid':[],'test':[]}
    subs = ['train','valid','test']
    subs = ['train','valid']
    fig1,ax=plt.subplots(1,3,figsize=(12,4))
    for ns,subset in enumerate(subs):
        mmin=20
        mmax=-20
        this_set = {'train':ds_train, 'valid':ds_val, 'test':ds_tst}[subset]
        with torch.no_grad():
            for n in range(len(this_set)):
                if 1:
                    Tmode = this_set[n][0]
                    if len(Tmode.shape) == 3:
                        Tmode = Tmode.squeeze(0)
                    moo = model( Tmode.unsqueeze(0).to(net.device))
                    if type(moo) == tuple:
                        moo = [mmm.cpu() for mmm in moo]
                        EB = this_set[n][1][0:1]
                    else:
                        moo = moo.cpu()
                        EB = this_set[n][1]
                    err[subset].append( model.criterion(moo,EB.unsqueeze(0)))
            errs = np.array([e.item() for e in err[subset]])
            args = np.argsort(errs)
            if subset == 'train':
                dothese = np.concatenate([np.arange(10),args[:10],args[-10:]])
            else:
                dothese = np.arange( len(this_set))
            for i,n in enumerate(dothese):
                Tmode = this_set[n][0]
                if len(Tmode.shape) == 3:
                    Tmode = Tmode.squeeze(0)
                moo = model( Tmode.unsqueeze(0).to(net.device))
                if type(moo) == tuple:
                    moo = moo[0].cpu()
                    EB = this_set[n][1]
                    Etarget = EB[0]
                    Eguess=moo[0][0].detach().numpy()
                else:
                    moo = moo.cpu()
                    EB = this_set[n][1][0]
                    Etarget = EB
                    Eguess=moo[0][0].detach().numpy()
                thiserr = errs[n]
                #plot_multipole.rmplot( sky[subset][n], rm, clm_model = moo, clm_real = clm, fname = "rm_and_sampled_%04d"%n)
                if 1:
                    import dtools_global.vis.pcolormesh_helper as pch
                    fig,axes=plt.subplots(2,3,figsize=(14,8))
                    ax0,ax1=axes
                    ppp = ax0[0].imshow(Tmode)
                    fig.colorbar(ppp,ax=ax0[0])
                    ax0[0].set(title='T')
                    Emin = min([Etarget.min(), Eguess.min()])
                    Emax = max([Etarget.max(), Eguess.max()])
                    Enorm = mpl.colors.Normalize(vmin=Emin,vmax=Emax)
                    ppp=ax0[1].imshow(Etarget,norm=Enorm)
                    fig.colorbar(ppp,ax=ax0[1])
                    ax0[1].set(title='E actual')
                    ax0[2].set(title='B actual')

                    ppp=ax1[1].imshow(Eguess,norm=Enorm)
                    fig.colorbar(ppp,ax=ax1[1])
                    ax1[1].set(title='E guess')
                    if moo.shape[1]>1 and False:
                        Btarget = EB[1]
                        Bguess=moo[0][1].detach().numpy()
                        Bmin = min([Btarget.min(), Bguess.min()])
                        Bmax = max([Btarget.max(), Bguess.max()])
                        Bnorm = mpl.colors.Normalize(vmin=Bmin,vmax=Bmax)
                        ppp=ax0[2].imshow(Btarget,norm=Bnorm)
                        fig.colorbar(ppp,ax=ax0[2])
                        ppp=ax1[2].imshow(Bguess,norm=Bnorm)
                        ax1[2].set(title='B guess')
                    fig.colorbar(ppp,ax=ax1[2])

                    E1 = Etarget.flatten()
                    E2 = Eguess.flatten()
                    ax1[0].set(title='%0.4f'%thiserr)
                    pch.simple_phase(E1,E2,ax=ax1[0])
                    mmin = E1.min()
                    mmax = E1.max()
                    ax1[0].plot( [mmin,mmax],[mmin,mmax],c='k')


                    fig.savefig('%s/plots/show_net%d_%s_%04d'%(os.environ['HOME'],model.idd,subset,i))
                    plt.close(fig)



        #this_err = torch.tensor(err[subset]).detach().numpy()
        ax[ns].hist(errs)
    fig1.tight_layout()
    oname = '%s/plots/errhist_net%d'%(os.environ['HOME'],model.idd)
    print(oname)
    fig1.savefig(oname)
    plt.close('all')


