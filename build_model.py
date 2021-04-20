from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from numpy.random import normal
from math import sqrt
import numpy as np
import math
import os.path as osp
from utils import *
import torch.nn as nn
import os.path as osp



class basic(nn.Module):

    def __init__(self):
        super(basic, self).__init__()

        self.add_module('conv1_1', nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
        self.add_module('relu1_1', nn.ReLU(inplace=True))   
        self.add_module('fc6', nn.Linear( 16*7*7, 1)) 

        self._initialize_weights()

    def forward(self, x):

        conv11_f = self.conv1_1(x)   
        conv11_f = self.relu1_1(conv11_f)
        
        vec = conv11_f.view(-1, 16*7*7)
        fc1=self.fc6(vec)
        fc2 = fc1
        fc2 = fc2.reshape(fc2.shape[0],fc2.shape[1],1,1)
        return fc2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            

