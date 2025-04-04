import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


# class BackBone_new(nn.Module):
#
#     def __init__(self,num_channel=64):
#         super().__init__()
#         ## 编码器
#         self.c1 = ConvBlock(3,num_channel)
#         self.ru1 = nn.ReLU(inplace=True)
#         self.pool = nn.MaxPool2d(2)
#         self.c2 = ConvBlock(num_channel,num_channel)
#         # nn.ReLU(inplace=True)
#         # nn.MaxPool2d(2)
#         self.c3 = ConvBlock(num_channel,num_channel)
#         # nn.ReLU(inplace=True)
#         # nn.MaxPool2d(2)
#         self.c4 = ConvBlock(num_channel,num_channel)
#         # nn.ReLU(inplace=True)
#         # nn.MaxPool2d(2)
#
#         ## 解码器
#         self.c5 = ConvBlock(128,num_channel)
#         self.ru2 = nn.ReLU(inplace=True)
#         self.c6 = ConvBlock(128, num_channel)
#         self.c7 = ConvBlock(128, num_channel)
#
#         self.tmp_conv = nn.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=8)
#
#     def forward(self,inp):
#         # return self.layers(inp)
#         ## 编码器
#         R1 = self.pool(self.ru1(self.c1(inp)))
#         R2 = self.pool(self.ru1(self.c2(R1)))
#         R3 = self.pool(self.ru1(self.c3(R2)))
#         R4 = self.pool(self.ru1(self.c4(R3)))
#
#         ## 解码器
#         o1 = self.ru2(self.c5(torch.cat((R3, self.upsample_1(R4)), dim=1)))
#         o2 = self.ru2(self.c6(torch.cat((R2, self.upsample_2(o1)), dim=1)))
#         o3 = self.ru2(self.c7(torch.cat((R1, self.upsample_1(o2)), dim=1)))
#
#         return self.tmp_conv(o3)
#
#     def upsample_1(self, feature_map):
#         return nn.functional.interpolate(feature_map, scale_factor=2, mode='bilinear', align_corners=False)
#
#     def upsample_2(self, feature_map):
#         return F.interpolate(feature_map, size=(21, 21), mode='bilinear', align_corners=False)

class BackBone_final(nn.Module):

    def __init__(self,num_channel=64):
        super().__init__()

        self.layer1 = nn.Sequential(ConvBlock(3,num_channel),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(ConvBlock(num_channel,num_channel),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(ConvBlock(num_channel,num_channel),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(ConvBlock(num_channel,num_channel),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(2))

    def forward(self, x):
        # return self.layers(inp)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        return x


class ConvBlock(nn.Module):
    
    def __init__(self,input_channel,output_channel):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_channel),
            EMA(output_channel))

    def forward(self,inp):
        return self.layers(inp)


class BackBone(nn.Module):

    def __init__(self,num_channel=64):
        super().__init__()
        
        self.layers = nn.Sequential(
            ConvBlock(3,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    def forward(self,inp):

        return self.layers(inp)
