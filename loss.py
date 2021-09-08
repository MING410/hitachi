import torch
from options import BaseOptions
import numpy as np
import cv2
import os
import time
import torch.nn as nn



def build_loss(mode='ce'):
    """Choices: ['ce' or 'focal']"""
    if mode == 'ce':
        return CrossEntropyLoss
    elif mode == 'focal':
        return FocalLoss
    else:
        raise NotImplementedError


def CrossEntropyLoss(logit, target,weight,ignore_index):
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                    reduction='mean')
    loss = criterion(logit, target.long())
    # if self.batch_average:
    #     loss /= n
    return loss


# logit=output
def FocalLoss(logit, target,weight,ignore_index,batch_average=True,gamma=2, alpha=0.5):
    # n, c, h, w = logit.size()
    n, c, h, w = logit.size()
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                    size_average=ignore_index)

    logpt = -criterion(logit, target.long())
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt

    if batch_average:
        loss /= n

    return loss