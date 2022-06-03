from __future__ import division
from turtle import forward

import torch
import torch.nn as nn


class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, pred, target, mask):
        n, c, h, w = pred.size()

        pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        pred = pred[mask.view(n * h * w, 1).repeat(1, c) == 1.]
        pred = pred.view(-1, c)

        target = target.transpose(1,2).transpose(2,3).contiguous().view(-1, c)
        target  = target[mask.view(n * h * w, 1).repeat(1, c) == 1.]
        target = target.view(-1,c)

        loss = self.pdist(pred, target)
        loss = torch.sum(loss, 0)
        loss /= mask.sum()

        return loss

class GaussianNLLLoss(nn.Module):
    def __init__(self, eps=1e-06):
        super().__init__()
        self.eps = eps
    
    def forward(self, target, mean, std, mask):
        n, c, h, w = target.size()

        target = target.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target[mask.view(n * h * w, 1).repeat(1, c) == 1.]
        target = target.view(-1, c)

        mean = mean.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        mean = mean[mask.view(n * h * w, 1).repeat(1, c) == 1.]
        mean = mean.view(-1, c)

        std = std.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        std = std[mask.view(n * h * w, 1).repeat(1, c) == 1.]
        var = torch.pow(std.view(-1, c), 2)

        loss = 0.5 * (torch.log(torch.clamp(var, min=self.eps)) + torch.pow(mean - target, 2) / torch.clamp(var, min=self.eps))
        loss = torch.sum(loss)
        loss /= mask.sum()

        return loss



class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, pred, target, mask):
        loss = self.celoss(pred, target)
        return (loss*mask).sum() / mask.sum()



    
        


