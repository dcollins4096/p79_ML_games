"""
Average of the trained net0600g (Q = T q) and net0600h (E,B output, converted back to Q,U).
No training: the two have partly independent errors, and on the Mach 4 test set the average beats
both (mean relMSE over Ma/los groups 0.547 vs 0.568 for g and 0.551 for h; sub-Alfvenic in-plane U
0.59 vs 0.64 / 0.62; r(k=64) close to g's).

Data, inputs and targets are 0600g's (targets T, Q, U).  thisnet() builds both models and loads
models/test0600g.pth and models/test0600h.pth.  For eval_scales.py / err_vs_params.py, save the
combined weights once with  python -m networks.net0600gh  (writes models/test0600gh.pth).
"""
import torch
import torch.nn as nn
import networks.net0600g as gnet
import networks.net0600h as hnet

idd = "0600gh"
what = "average of the 0600g and 0600h predictions (no training)"

fname_train = gnet.fname_train
fname_test = gnet.fname_test
load_data = gnet.load_data
input_channels = gnet.input_channels      # T, velocity centroid, velocity variance, projected |B|
SphericalDataset = gnet.SphericalDataset  # targets T, Q, U
set_seed = gnet.set_seed
device = gnet.device
assert hnet.input_channels == gnet.input_channels

class main_net(nn.Module):
    def __init__(self, g, h, tta=True):
        super().__init__()
        self.g, self.h = g, h
        self.tta = tta
        # 0600h's input features are 0600g's 13 plus E,B of T; the generative models condition on them
        self.in_norm = h.in_norm
        self.train_curve, self.val_curve = g.train_curve, g.val_curve   # for plotter.plot_loss_curve

    def input_features(self, x):
        return self.h.input_features(x)

    def forward(self, x):
        """x [B,4,H,W] -> (prediction [B,3,H,W] = T, Q, U,)"""
        if x.ndim == 3:
            x = x.unsqueeze(1)
        self.g.tta = self.h.tta = self.tta
        pg = self.g(x)[0]
        ph = hnet.eb_to_qu(self.h(x)[0])
        return (0.5 * (pg + ph),)

    def criterion1(self, preds, target):
        """Diagnostics only (for plotter.plot1): the main 0600g loss terms on the averaged prediction"""
        p, t = preds[0][:, 1:], target[:, 1:]
        return {'L1_0': nn.functional.l1_loss(p, t),
                'Pear': (gnet.pearson_loss(p[:, 0:1], t[:, 0:1]) + gnet.pearson_loss(p[:, 1:2], t[:, 1:2])) / 2,
                'Rk': gnet.rk_loss(p, t)}

def thisnet():
    g = gnet.thisnet()
    g.load_state_dict(torch.load("models/test0600g.pth", map_location=device))
    h = hnet.thisnet()
    h.load_state_dict(torch.load("models/test0600h.pth", map_location=device))
    model = main_net(g, h).to(device)
    model.eval()
    return model

def train(model, all_data):
    print("net0600gh is an average of two trained models; nothing to train")

if __name__ == '__main__':
    torch.save(thisnet().state_dict(), "models/test0600gh.pth")
    print("saved models/test0600gh.pth")
