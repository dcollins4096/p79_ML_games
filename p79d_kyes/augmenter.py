
import torch


def aug(x,y):
    """
    Randomly flip and rotate a whole batch the same way.
    x: [B, C, N, N]
    returns: [B, C, N, N]
    """
    #if torch.randint(0,3) > 7:
    #    return x,y

    # Random rotation (0, 90, 180, 270)
    k = torch.randint(0, 4, (1,)).item()
    x = torch.rot90(x, k, dims=[2, 3])  # rotate over H,W
    y = torch.rot90(y, k, dims=[2, 3])  # rotate over H,W

    # Random horizontal flip
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[3])  # flip W
        y = torch.flip(y, dims=[3])  # flip W


    # Random vertical flip
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[2])  # flip H
        y = torch.flip(y, dims=[2])  # flip H

    return x, y

