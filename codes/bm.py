# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:52:59 2020

@author: sujitk
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variables

#movies = pd.read_csv('ml-lm/movies.dat',sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#movies = pd.read_csv('ml-lm/users.dat',sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#ratings = pd.read_csv('ml-lm/ratings.dat',sep = '::', header = None, engine = 'python', encoding = 'latin-1')

training_set = pd.read_csv("~/Documents/net_prop.csv")
#net_prop = pd.read_csv("~/Documents/net_prop.csv")

training_set = np.array(training_set, dtype = 'int')
#test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
#test_set = np.array(test_set, dtype = 'int')

class RBM():
    def __init__(self,nv,nh):
        self.W = torch.randn(nh,nv) 
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_v = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self,v0, vk, ph0, phk):
        self.W += torch.mm(v0.t() - torch.mm(vt.t(),phk))
        self.b += torch.sum((v0 - vk),0)
        self.a += torch.sum((ph0 - phk), 0)
    
nv = len(training_set)
nh = 100
batch_size = 35
rbm = RBM(nv,nh)
    
#training the RBM
nb_epoch = 10
for epoch in (1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for counties in range(0, nb_counties - batch_size, batch_size):
        vk = training_set[counties:counties+batch_size]
        v0 = training_set[counties:counties+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss += torch.mean(torch.abs(v0 - vk))
        s += 1.
    print('epoch: '+ str(epoch)+ ' loss: '+str(train_loss/s))

            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
