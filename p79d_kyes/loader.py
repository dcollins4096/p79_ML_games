

import h5py
import torch

def loader(fname,ntrain=1, nvalid=1):
    if nvalid != 30:
        print("nvalid forced to be the last 30")
    fptr = h5py.File(fname,'r')
    subsets = fptr['subsets'][()]
    subsets = torch.tensor(subsets,dtype=torch.float32)
    #output={'train':subsets[:ntrain], 'valid':subsets[ntrain:ntrain+nvalid],'test':subsets[ntrain+nvalid:]}
    output={'train':subsets[:ntrain], 'valid':subsets[-30:],'test':subsets[ntrain:-30]}
    return output

