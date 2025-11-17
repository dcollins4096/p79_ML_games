from importlib import reload
import importlib
import torch
import plotter
reload(plotter)

def correlations(net_name,**kwargs):
    net = importlib.import_module(f"networks.{net_name}")
    suffix = ""
    for key in sorted(kwargs):
        suffix += "%s=%s"%(key, kwargs[key])
    kwargs['suffix']=suffix
    oname = "models/%s_%s"%(net_name, suffix)
    model = net.main_net(**kwargs)
    model.load_state_dict(torch.load(oname))
    model = model.to('cuda')
    model.eval()
    all_data = net.load_data()
    error_list, rrr = plotter.correlations(net_name,model, all_data, suffix)
    return error_list, rrr, suffix

def ploot(net_name,**kwargs):
    net = importlib.import_module(f"networks.{net_name}")
    suffix = ""
    for key in sorted(kwargs):
        suffix += "%s=%s"%(key, kwargs[key])
    kwargs['suffix']=suffix
    oname = "models/%s_%s"%(net_name, suffix)
    print("LOAD",oname)
    model = net.main_net(**kwargs)
    model.load_state_dict(torch.load(oname))
    model = model.to('cuda')
    model.eval()
    all_data = net.load_data()
    return plotter.plot(net_name,model, all_data, suffix, erronly=False)

def go(net_name, **kwargs):
    net = importlib.import_module(f"networks.{net_name}")
    reload(net)
    suffix = ""
    for key in sorted(kwargs.keys()):
        suffix += "%s=%s"%(key, kwargs[key])

    all_data = net.load_data()
    kwargs['suffix']=suffix
    model = net.main_net(**kwargs)
    model.idd = net.idd
    net.train(model,all_data)

    oname = "models/%s_%s"%(net_name, suffix)
    torch.save(model.state_dict(), oname)
    print("model saved ",oname)

    plotter.plot(net_name, model, all_data, suffix)
    return model


