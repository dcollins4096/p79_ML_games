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
import torch_power
reload(loader)
reload(torch_power)

if not os.path.exists('dtools/starter1.py'):
    print("Need to get the submodule")
    print("git submodule init")
    print("git submodule update")
    sys.exit(-1)


new_model   = 1
load_model  = 0
train_model = 1
save_model = 1
plot_models = 1

def nparam(model):
    return sum( param.numel() for param in model.parameters() if param.requires_grad)

if new_model:
    import networks.net0600  as net
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

