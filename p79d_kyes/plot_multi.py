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
sys.path.append('/home/dcollins/repos/')
import dtools_global.vis.pcolormesh_helper as pch
from scipy.stats import pearsonr
import tqdm
import torch_power
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from matplotlib.lines import Line2D
import dtools_global.math.power_spectrum as ps

import networks.net0699 as net99

net_list = [184,607,605, 608]

def compute_losses(model,all_data, suite='test'):
    this_set = loaders[suite]
    error_list = defaultdict(list)
    with torch.no_grad():
        for xb,yb in tqdm.tqdm(this_set):
            xb = xb.to(device)
            yb = yb.to(device)
            moo = model( xb )

            err_dict= model.criterion1(moo,yb)
            for err in err_dict:
                error_list[err].append(err_dict[err].item())
            error_list['total'].append( sum(err_dict.values()).item())
    return error_list

all_data = net99.load_data()
ds_train = net99.SphericalDataset(all_data['train'])
ds_val   = net99.SphericalDataset(all_data['valid'])
ds_tst   = net99.SphericalDataset(all_data['test'][:100])

train_loader = DataLoader(ds_train, batch_size=1, shuffle=False, drop_last=False)
val_loader   = DataLoader(ds_val,   batch_size=1, shuffle=False, drop_last=False)
tst_loader   = DataLoader(ds_tst,   batch_size=1, shuffle=False, drop_last=False)
loaders={'test':tst_loader,'train':train_loader,'valid':val_loader}

device = 'cuda'

if 'net_dict' not in dir():
    net_dict={}
    model_dict={}
    error_dict = {}
    for net_idd in net_list:
        net_name = 'net%04d'%net_idd
        net = importlib.import_module(f"networks.{net_name}")
        model = net.thisnet()
        model = model.to(device)
        model.load_state_dict(torch.load("models/test%d.pth"%net.idd, map_location=torch.device('cpu')))
        net_dict[net_idd]=net
        model_dict[net_idd] = model
        error_dict[net_idd] = compute_losses(model_dict[net_idd], all_data)



linestyles = ['-','--','-.',':']
if 1:
    fig,ax=plt.subplots(1,1)
    colors = {'L1_0':'c','L1_1':'m','L1_2':'y','L1_3':'m','L1_4':'m','L1_Multi':'m','Pear':'r','Grad':'g','SSIM':'b','Power':'purple', 'total':'k'}
    for nnet, net_idd in enumerate(net_list):
        error_list = error_dict[net_idd]
        for err in error_list:
            eee =  error_list[err]
            x = sorted(eee)
            y = np.arange(len(x))/len(x)
            ax.plot(x,y,linestyle=linestyles[nnet],color=colors.get(err,'orange'))
    color_handles = []
    color_labels = []
    for err in error_list:
        color_handles.append( Line2D([0],[0],color=colors.get(err,'orange')))
        color_labels.append(err)
    legend1 = plt.legend(color_handles, color_labels, loc='upper left')
    line_handles = []
    for nnet, net_idd in enumerate(net_list):
        line_handles.append( Line2D([0],[0], linestyle=linestyles[nnet]))
    legend2 = plt.legend(line_handles, net_list, loc='lower left')
    plt.gca().add_artist(legend1)


    ax.set(xscale='log',xlim=[1e-2,10])
    fig.savefig('%s/plots/multi_err.png'%(os.environ['HOME']))

if 1:
    ncol = 2+len(net_list)
    this_set = loaders['valid']

    with torch.no_grad():
        count=-1
        for xb,yb in tqdm.tqdm(this_set):
            count+=1
            if count > 2:
                break
            fig,ax=plt.subplots(3,ncol)
            xb = xb.to(device)
            yb = yb.to(device)
            EB = yb[0]
            ax[0][0].set(title='fid')
            for n in range(3):
                Etarget = EB[n].cpu()
                ax[n][0].imshow(Etarget)
                power_target = ps.powerspectrum(Etarget.detach().numpy())
                ax[n][-1].plot( power_target.kcen, power_target.avgpower,c='k')
                ax[n][-1].set(xscale='log',yscale='log')
                ax[n][0].set_xticks([])
                ax[n][0].set_yticks([])
                ax[n][-1].set_xticks([])
                ax[n][-1].set_yticks([])

            for nnet,net_idd in enumerate(net_list):
                model = model_dict[net_idd]
                moo = model( xb )
                ax[0][nnet+1].set(title='%d'%net_idd)

                for n in range(3):
                    Eguess = moo[0][0][n].cpu()
                    Etarget = EB[n].cpu()
                    ax[n][nnet+1].imshow(Eguess)
                    cross=ps.cross_spectrum(Eguess.detach().numpy(), Etarget.detach().numpy())
                    ax[n][-1].plot(cross.kcen, cross.avgpower)#, c='r')
                    ax[n][nnet+1].set_xticks([])
                    ax[n][nnet+1].set_yticks([])


            fig.tight_layout()
            fig.savefig('%s/plots/multi_image_%04d'%(os.environ['HOME'], count))
            


