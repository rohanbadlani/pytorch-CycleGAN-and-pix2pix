import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb

def normalize_batch(batch):
    # normalize using imagenet mean and std. Implementation adapted from https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    # pdb.set_trace()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda()

    p2_norm = (batch + 1)/2 # [-1, 1] => [0, 1]
    
    p2_norm = (p2_norm - mean)/std

    return p2_norm

def perceptual_loss(real_features, fake_features, norm_variant=1):
    loss = None
    if norm_variant == 1:
        # L1 loss
        loss = nn.L1Loss()
    else:
        # L2 loss
        loss = nn.MSELoss()
    return loss(fake_features, real_features)