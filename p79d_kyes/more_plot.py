

def correlations(net_name,model, all_data, suffix):
    do_bispectrum=True
    net = importlib.import_module(f"networks.{net_name}")

    plot_loss_curve(net_name, model, suffix)

    ds_train = net.SphericalDataset(all_data['train'].to('cuda'))
    ds_val   = net.SphericalDataset(all_data['valid'].to('cuda'), rotation_prob=0.0)
    ds_tst   = net.SphericalDataset(all_data['test'].to('cuda'))
    model = model.to('cuda')

    this_set = ds_tst

    error_list = defaultdict(list)
    rrr = {'TE':[],'TB':[],'EB':[]}
    with torch.no_grad():
        for n in tqdm.tqdm(range(len(this_set))):
            if n >100:
                if 'kludge' not in dir():
                    kludge = True
                    print("KLUDGE for speed")
                continue
            Tmode = this_set[n][0]
            if len(Tmode.shape) == 3:
                Tmode = Tmode.squeeze(0)
            moo = model( Tmode.unsqueeze(0).to(net.device))
            if type(moo) == tuple:
                #moo = [mmm.cpu() for mmm in moo]
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

            err_dict= model.criterion1(moo,EB.unsqueeze(0).to(net.device))
            for err in err_dict:
                error_list[err].append(err_dict[err].item())
            error_list['total'].append( sum(err_dict.values()).item())

            T = EB[0].flatten().cpu().detach().numpy()
            E = EB[1].flatten().cpu().detach().numpy()
            B = EB[2].flatten().cpu().detach().numpy()
            rrr['TB'].append( pearsonr( T, B)[0])
            rrr['TE'].append( pearsonr( T, E)[0])
            rrr['EB'].append( pearsonr( E, B)[0])
    for q in ["Ma_act", "Ma_mean", "Ms_act", "Ms_mean", "frame", "los"]:
        error_list[q]= all_data['quantities']['test'][q]

    return error_list, rrr

def plot_corr(net_name,error_list, rrr, suffix):
    ncol = 4
    nrow = len(error_list.keys())//ncol
    if len(error_list.keys()) %ncol: nrow+=1
    for tt in ['TE','TB','EB']:
        fig,axes=plt.subplots(nrow,ncol, figsize=(12,12))
        aaa = axes.flatten()
        for nr,er in enumerate(error_list):
            aaa[nr].scatter( rrr[tt], error_list[er] )
            aaa[nr].set(xlabel=tt, ylabel=er)
        fig.tight_layout()
        fig.savefig('%s/plots/%s_%s_play_%s.png'%(os.environ['HOME'], net_name, suffix,tt))
        plt.close(fig)
    fig,axes=plt.subplots(2,3)
    aaa=axes.flatten()
    for nq,q in enumerate(['Ms_mean','Ms_act','Ma_mean','Ma_act','los','frame']):
        aaa[nq].scatter( error_list['Power'], error_list[q])
        aaa[nq].set(xlabel='Power',ylabel=q)
    fig.tight_layout()
    fig.savefig('%s/plots/%s_%s_power.png'%(os.environ['HOME'], net_name, suffix))


def plot_hot(net_name,model, all_data, suffix, erronly=True):

    net = importlib.import_module(f"networks.{net_name}")

    plot_loss_curve(net_name, model, suffix)

    ds_train = net.SphericalDataset(all_data['train'].to('cuda'))
    ds_val   = net.SphericalDataset(all_data['valid'].to('cuda'))
    ds_tst   = net.SphericalDataset(all_data['test'].to('cuda'))
    model = model.to('cuda')

    #this_set = ds_val
    if 0:
        this_set = ds_train
        subs=['train']
    if 1:
        this_set = ds_tst
    this_set=ds_val
    subs=['valid']

    error_list = defaultdict(list)
    with torch.no_grad():
        for n in tqdm.tqdm(range(len(this_set))):
            if n ==-1:
                if 'kludge' not in dir():
                    kludge = True
                    print("KLUDGE for speed")
                continue
            Tmode = this_set[n][0]
            if len(Tmode.shape) == 3:
                Tmode = Tmode.squeeze(0)
            out_prob = model( Tmode.unsqueeze(0).to(net.device))

            EB = this_set[n][1] #include all three, criterion removes the first

            error= model.criterion(out_prob,EB.unsqueeze(0).to(net.device))
            error_list['total'].append(error.item())
    fig,ax=plt.subplots(1,1)
    x = sorted(error_list['total'])
    y = np.arange(len(x))/len(x)
    ax.plot(x,y)
    ax.set(xscale='log',xlim=[1e-2,5])
    fig.savefig('%s/plots/%s_%s_errors.png'%(os.environ['HOME'],net_name,suffix))

    for ns,subset in enumerate(subs):
        this_set = {'train':ds_train, 'valid':ds_val, 'test':ds_tst}[subset]
        if subset == 'test' or subset=='train':
            ppp = np.argsort(error_list['total'])
            #dothese = list( ppp[::ppp.size//30])
            dothese = list(ppp[:2])+list(ppp[-2:])
        else:
            pass
        
        dothese = range(len(this_set))
        #dothese = range(len(this_set))
        #dothese = [0]
        print("DO", dothese)

        with torch.no_grad():
            for i,n in tqdm.tqdm(enumerate(dothese)):
                Tmode = this_set[n][0]
                if len(Tmode.shape) == 3:
                    Tmode = Tmode.squeeze(0)
                out_main, out_prob = model( Tmode.unsqueeze(0).to(net.device))
                mean, var = model.meanie(out_prob)
                EB = this_set[n][1][1:]
                #EBguess = mean.squeeze(0)
                EBguess = out_main[0,1:,...]
                var = var.squeeze(0)

                if 1:
                    fig,axes=plt.subplots(3,6,figsize=(15,8))
                    ax0,ax1,ax2=axes
                    for aaa in ax0:
                        aaa.imshow(Tmode.detach().cpu().numpy())
                    plot_three_err(EBguess[0,:,:], EB[0,:,:],var[0,:,:],out_prob[0,0,...],mean[0,0,...],model,title='E', axs=ax1, fig=fig)
                    plot_three_err(EBguess[1,:,:], EB[1,:,:],var[1,:,:],out_prob[0,1,...],mean[0,1,...],model,title='B', axs=ax2, fig=fig)

                    fig.tight_layout()
                    fig.savefig('%s/plots/%s_%s_%s_%04d.png'%(os.environ['HOME'],net_name,suffix,subset,i))
                    plt.close(fig)

    return model


def correlations(net_name,model, all_data, suffix):
    do_bispectrum=True
    net = importlib.import_module(f"networks.{net_name}")

    plot_loss_curve(net_name, model, suffix)

    ds_train = net.SphericalDataset(all_data['train'].to('cuda'))
    ds_val   = net.SphericalDataset(all_data['valid'].to('cuda'), rotation_prob=0.0)
    ds_tst   = net.SphericalDataset(all_data['test'].to('cuda'))
    model = model.to('cuda')

    this_set = ds_tst

    error_list = defaultdict(list)
    rrr = {'TE':[],'TB':[],'EB':[]}
    with torch.no_grad():
        for n in tqdm.tqdm(range(len(this_set))):
            if n >100:
                if 'kludge' not in dir():
                    kludge = True
                    print("KLUDGE for speed")
                continue
            Tmode = this_set[n][0]
            if len(Tmode.shape) == 3:
                Tmode = Tmode.squeeze(0)
            moo = model( Tmode.unsqueeze(0).to(net.device))
            if type(moo) == tuple:
                #moo = [mmm.cpu() for mmm in moo]
                nchannels = moo[0].shape[1]
                if nchannels == 2 or nchannels == 3:
                    EB = this_set[n][1]
                else:
                    EB = this_set[n][1][0:1]
            else:
                moo = moo.cpu()
                this_set = this_set.cpu()
                if moo[0].shape[1] == 2:
                    EB = this_set[n]
                else:
                    EB = this_set[n][1]

            err_dict= model.criterion1(moo,EB.unsqueeze(0).to(net.device))
            for err in err_dict:
                error_list[err].append(err_dict[err].item())
            error_list['total'].append( sum(err_dict.values()).item())

            T = EB[0].flatten().cpu().detach().numpy()
            E = EB[1].flatten().cpu().detach().numpy()
            B = EB[2].flatten().cpu().detach().numpy()
            rrr['TB'].append( pearsonr( T, B)[0])
            rrr['TE'].append( pearsonr( T, E)[0])
            rrr['EB'].append( pearsonr( E, B)[0])
    for q in ["Ma_act", "Ma_mean", "Ms_act", "Ms_mean", "frame", "los"]:
        error_list[q]= all_data['quantities']['test'][q]

    return error_list, rrr

def plot_corr(net_name,error_list, rrr, suffix):
    ncol = 4
    nrow = len(error_list.keys())//ncol
    if len(error_list.keys()) %ncol: nrow+=1
    for tt in ['TE','TB','EB']:
        fig,axes=plt.subplots(nrow,ncol, figsize=(12,12))
        aaa = axes.flatten()
        for nr,er in enumerate(error_list):
            aaa[nr].scatter( rrr[tt], error_list[er] )
            aaa[nr].set(xlabel=tt, ylabel=er)
        fig.tight_layout()
        fig.savefig('%s/plots/%s_%s_play_%s.png'%(os.environ['HOME'], net_name, suffix,tt))
        plt.close(fig)
    fig,axes=plt.subplots(2,3)
    aaa=axes.flatten()
    for nq,q in enumerate(['Ms_mean','Ms_act','Ma_mean','Ma_act','los','frame']):
        aaa[nq].scatter( error_list['Power'], error_list[q])
        aaa[nq].set(xlabel='Power',ylabel=q)
    fig.tight_layout()
    fig.savefig('%s/plots/%s_%s_power.png'%(os.environ['HOME'], net_name, suffix))


    
    
