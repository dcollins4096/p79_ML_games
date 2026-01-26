import torch
import random
import numpy as np
import os

def set_seed(seed=8675309):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)   # strongest setting
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False     # IMPORTANT
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
