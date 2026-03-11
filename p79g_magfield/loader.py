

import h5py
import torch
import os
import pdb
import bucket

def loader(fname,ntrain=1, nvalid=1):
    full_name = "datasets/"+fname
    if full_name in bucket.stuff:
        return bucket.stuff[full_name]
    if not os.path.exists(full_name):
        print("File not found. %s"%full_name)
        print(" Fetch it and then press c for continue")
        print(" scp cloudbreak:/data/cb1/Projects/P79d_ML/%s ./datasets"%full_name)
        pdb.set_trace()
    fptr = h5py.File(full_name,'r')
    subsets = fptr['subsets'][()]
    subsets = torch.tensor(subsets,dtype=torch.float32)
    #output={'train':subsets[:ntrain], 'valid':subsets[ntrain:ntrain+nvalid],'test':subsets[ntrain+nvalid:]}
    output={'train':subsets[:ntrain], 'valid':subsets[-nvalid:],'test':subsets[ntrain:-nvalid]}
    if 'Ma_act' in fptr:
        quantities={'train':{},'valid':{}, 'test':{}}
        for q in ["Ma_act", "Ma_mean", "Ms_act", "Ms_mean", "frame", "los"]:
            qqq = fptr[q]
            quantities['train'][q] = qqq[:ntrain]
            quantities['valid'][q] = qqq[-nvalid:]
            quantities['test'][q]  = qqq[ntrain:-nvalid]

        output['quantities'] = quantities

    fptr.close()
    bucket.stuff[full_name] = output
    return output

