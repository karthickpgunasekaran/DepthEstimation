import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_


class PoseDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1, bias=True),
                                   nn.ELU(inplace=True)
                                   )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256*3, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ELU(inplace=True)
        )

        self.conv3 =  nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
        nn.ELU(inplace=True),
        nn.Conv2d(256, 12, kernel_size=3,stride =2,padding=1, bias=True),
        )

    def forward(self, d1,d2,d3):
        pconv0_t = self.conv1(d2)
        pconv0_s1 = self.conv1(d1)
        pconv0_s2 = self.conv1(d3)
        pconv1 = torch.cat((pconv0_s1,pconv0_t,pconv0_s2),dim=1)
        pconv2 = self.conv2(pconv1)
        pose= self.conv3(pconv2)
        batch, c, h, w = pose.size()
        pose = pose.view(batch, 2, 6) #*(10**-2)
        trans = pose[:, :, :3] * 0.001
        rot = pose[:, :, 3:] * 0.01
        #pose[:, :, :3] = pose[:, :, :3] * 0.001
        #pose[:, :, 3:] = pose[:, :, 3:] * 0.01
        return pose, trans, rot
