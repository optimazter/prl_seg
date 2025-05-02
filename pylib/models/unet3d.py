import torch
from torch import nn

#https://debuggercafe.com/unet-from-scratch-using-pytorch/
#https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint




class UNet3d(nn.Module):

    def __init__(self, n_channels, n_classes, use_checkpoint=True):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv = self.build_double_conv(n_channels, 64)

        self.down1 = self.build_down_conv(64, 128)
        self.down2 = self.build_down_conv(128, 256)
        self.down3 = self.build_down_conv(256, 512)
        self.down4 = self.build_down_conv(512, 1024)

        self.up1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2) 
        self.conv1 = self.build_double_conv(1024, 512)
        self.up2 =nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv2 = self.build_double_conv(512, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv3 = self.build_double_conv(256, 128)
        self.up4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv4 = self.build_double_conv(128, 64)

        self.head = nn.Conv3d(64, n_classes, kernel_size=1)


    def pad(self, x1, x2):
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        return F.pad(x1, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2,
                          diffZ // 2, diffZ - diffZ // 2])

    def build_down_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool3d(2),
            self.build_double_conv(in_channels, out_channels)
        )

    def build_double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        d1 = self.in_conv(x)
        if self.use_checkpoint:
            d2 = checkpoint(self.down1, d1)
            d3 = checkpoint(self.down2, d2)
            d4 = checkpoint(self.down3, d3)
            d5 = checkpoint(self.down4, d4)
        else:
            d2 = self.down1(d1)
            d3 = self.down2(d2)
            d4 = self.down3(d3)
            d5 = self.down4(d4)

        if self.use_checkpoint:
            u1 = checkpoint(self.up1, d5)
            u1 = self.pad(u1, d4)
            x = torch.cat([d4, u1], dim=1)
            x = checkpoint(self.conv1, x)
        else:
            u1 = self.up1(d5)
            u1 = self.pad(u1, d4)

            x = torch.cat([d4, u1], dim=1)
            x = self.conv1(x)

        if self.use_checkpoint:
            u2 = checkpoint(self.up2, x)
            u2 = self.pad(u2, d3)
            x = torch.cat([d3, u2], dim=1)
            x = checkpoint(self.conv2, x)
        else:
            u2 = self.up2(x)
            u2 = self.pad(u2, d3)
            x = torch.cat([d3, u2], dim=1)
            x = self.conv2(x)
        
        if self.use_checkpoint:
            u3 = checkpoint(self.up3, x)
            u3 = self.pad(u3, d2)
            x = torch.cat([d2, u3], dim=1)
            x = checkpoint(self.conv3, x)
        else:
            u3 = self.up3(x)
            u3 = self.pad(u3, d2)
            x = torch.cat([d2, u3], dim=1)
            x = self.conv3(x)

        if self.use_checkpoint:
            u4 = checkpoint(self.up4, x)
            u4 = self.pad(u4, d1)
            x = torch.cat([d1, u4], dim=1)
            x = checkpoint(self.conv4, x)
        else:
            u4 = self.up4(x)
            u4 = self.pad(u4, d1)
            x = torch.cat([d1, u4], dim=1)
            x = self.conv4(x)

        out = self.head(x)
        return out


    
        
