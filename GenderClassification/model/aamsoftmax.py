#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print(target,correct,target) #1, 0,0 ... / False, True, .... / 0, 0, 0 ...

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.1, scale=15, easy_margin=False, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        #print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

    def forward(self, x, label=None):
        #print("Loss", x.shape, label.size()[0], self.in_feats)
        #print(x.shape) #32, 512 => 10,512(test)
        assert x.size()[0] == label.size()[0] #32, 3, 256
        assert x.size()[1] == self.in_feats
        # cos(theta)
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        #print(one_hot, label.view(-1, 1))

        # (10,3) -> (30,1)


        one_hot.scatter_(1, label.view(-1, 1), 1)
        #print(one_hot)
        #print("one_hot1", one_hot.shape) # 10, 256
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        #print(output, label)
        #print("one_hot2", one_hot.shape) # 10 , 3


        loss    = self.ce(output, label)
        prec1, pred = accuracy(output.detach(), label.detach(), topk=(1,))
        prec1 = prec1[0]
        pred = pred[0]

        
        return loss, prec1, output, pred