from importlib import reload
import torch
import networks.net0141 as net
reload(net)
import plotter
reload(plotter)

def ploot(**kwargs):
    suffix = ""
    for key in kwargs:
        suffix += "%s=%s"%(key, kwargs[key])
    kwargs['suffix']=suffix
    oname = "models/net%04d_%s"%(net.idd, suffix)
    model = net.main_net(**kwargs)
    model.load_state_dict(torch.load(oname))
    model = model.to('cuda')
    all_data = net.load_data()
    plotter.plot(model, all_data, suffix)

def go(**kwargs):
    suffix = ""
    for key in sorted(kwargs.keys()):
        suffix += "%s=%s"%(key, kwargs[key])

    all_data = net.load_data()
    kwargs['suffix']=suffix
    model = net.main_net(**kwargs)
    model.idd = net.idd
    net.train(model,all_data)

    oname = "models/net%04d_%s"%(net.idd, suffix)
    torch.save(model.state_dict(), oname)
    print("model saved ",oname)

    plotter.plot(model, all_data, suffix)


