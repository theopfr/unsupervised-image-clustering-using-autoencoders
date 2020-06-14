

import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        """ encoder """
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5))
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=3)
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=3)
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.maxpool2x2 = nn.MaxPool2d(2)   # not in usage

        """ target-decoder """
        self.upsample2x2 = nn.Upsample(scale_factor=2)   # not in usage

        self.targetDeconv1 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=3)
        self.targetBatchnorm1 = nn.BatchNorm2d(64)

        self.targetDeconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=3)
        self.targetBatchnorm2 = nn.BatchNorm2d(32)

        self.targetDeconv3 = nn.ConvTranspose2d(32, 3, kernel_size=(5, 5))
        self.targetBatchnorm3 = nn.BatchNorm2d(3)
    
    def _gaussian_noise_layer(self, x, fixed_gaussian_noise_rate=False):
        gaussian_noise_rate = np.random.choice([0.02, 0.05, 0.03, 0.0, 0.01, 0.02, 0.005])
        if fixed_gaussian_noise_rate is not False:
            gaussian_noise_rate = fixed_gaussian_noise_rate

        x += gaussian_noise_rate * torch.randn(1, 32, 32).cuda()

        return x

    def forward(self, x, train_: bool=True, print_: bool=False, return_bottlenecks: bool=False):
        if train_:
            x = self._gaussian_noise_layer(x)

        if print_: print(x.shape)

        """ encoder """
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)

        if print_: print(x.shape)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)

        if print_: print(x.shape)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        bottlenecks = F.relu(x)

        if print_: print(bottlenecks.shape)

        """ decoder """
        x = self.targetDeconv1(bottlenecks)
        x = self.targetBatchnorm1(x)
        x = F.relu(x)

        if print_: print(x.shape)

        x = self.targetDeconv2(x)
        x = self.targetBatchnorm2(x)
        x = F.relu(x)

        if print_: print(x.shape)

        x = self.targetDeconv3(x)
        x = torch.sigmoid(x)

        if print_: print(x.shape)

        if return_bottlenecks:
            return bottlenecks
        return x



"""x = torch.Tensor(torch.rand((1, 3, 32, 32))).cuda()
model = Model().cuda()
x = model.forward(x, print_=True)"""

