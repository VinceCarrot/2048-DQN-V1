import torch
import torch.nn as nn
import numpy as np
import math

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(17, 128, kernel_size=2, stride=1, padding=0), nn.LeakyReLU(0.1, inplace=True))
        self.conv12 = nn.Sequential(nn.Conv2d(128, 512, kernel_size=2, stride=1, padding=0), nn.LeakyReLU(0.1, inplace=True))
        self.conv13 = nn.Sequential(nn.Conv2d(512, 2048, kernel_size=2, stride=1, padding=0), nn.LeakyReLU(0.1, inplace=True))
        self.conv20 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True))
        self.conv21 = nn.Sequential(nn.Conv2d(32, 128, kernel_size=2, stride=1, padding=0), nn.LeakyReLU(0.1, inplace=True))
        self.conv22 = nn.Sequential(nn.Conv2d(128, 512, kernel_size=2, stride=1, padding=0), nn.LeakyReLU(0.1, inplace=True))
        self.conv23 = nn.Sequential(nn.Conv2d(512, 2048, kernel_size=2, stride=1, padding=0), nn.LeakyReLU(0.1, inplace=True))
        
        self.conv3 = nn.Sequential(nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(0.1, inplace=True))
        self.conv4 = nn.Conv2d(4096, 5, kernel_size=1, stride=1, padding=0)
        # self._create_weights()

        self.models = nn.ModuleList()
        self.models.append(self.conv11[0])
        self.models.append(self.conv12[0])
        self.models.append(self.conv13[0])
        self.models.append(self.conv20[0])
        self.models.append(self.conv21[0])
        self.models.append(self.conv22[0])
        self.models.append(self.conv23[0])
        self.models.append(self.conv3[0])
        self.models.append(self.conv4)


    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, px, nx):
        
        x1 = self.conv11(px)
        x1 = self.conv12(x1)
        x1 = self.conv13(x1)

        x2 = self.conv20(nx)
        x2 = self.conv21(x2)
        x2 = self.conv22(x2)
        x2 = self.conv23(x2)
        # print x1.size(), x2.size()
        xc = torch.cat((x1,x2),1)
        xc = self.conv3(xc)
        xc = self.conv4(xc)
        # print xc.size()
        
        return xc

    def CheckMaxMinParam(self):
        minV = float('Inf')
        maxV = float('-Inf')
        for m in range(len(self.models)):
            maxW = torch.max(self.models[m].weight)
            minW = torch.min(self.models[m].weight)
            if maxW > maxV:
                maxV = maxW
            if minW < minV:
                minV = minW

        return maxV, minV

