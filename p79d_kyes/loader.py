

import h5py
import torch
import pdb

def loader(fname,ntrain=1, nvalid=1):
    fptr = h5py.File(fname,'r')
    subsets = fptr['subsets'][()]
    subsets = torch.tensor(subsets,dtype=torch.float32)
    #output={'train':subsets[:ntrain], 'valid':subsets[ntrain:ntrain+nvalid],'test':subsets[ntrain+nvalid:]}
    output={'train':subsets[:ntrain], 'valid':subsets[-30:],'test':subsets[ntrain:-30]}
    if 'Ma_act' in fptr:
        quantities={'train':{},'valid':{}, 'test':{}}
        for q in ["Ma_act", "Ma_mean", "Ms_act", "Ms_mean", "frame", "los"]:
            qqq = fptr[q]
            quantities['train'][q] = qqq[:ntrain]
            quantities['valid'][q] = qqq[-30:]
            quantities['test'][q]  = qqq[ntrain:-30]

        output['quantities'] = quantities

    fptr.close()
    return output

