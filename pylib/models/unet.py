import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F




class PRLUNet(nn.Module):

    """
    A PyTorch implementation of the dual encoder U-Net, PRLU-Net by Adrian Hjertholm Voldeth (2025). 
    Based on the original U-Net architecture by Ronneberger et al. (2015).
    The code is also inspired by the implementation from Sovit Ranjan RathSovit Ranjan Rath (2023) available at: http://debuggercafe.com/unet-from-scratch-using-pytorch/, and 
    the code by milesal: https://github.com/milesial/Pytorch-UNet/tree/master?tab=readme-ov-file.

    Reference:
        Voldeth, A. H. (2025). AI Driven Paramagnetic Rim Lesion Differentiation in Multiple Sclerosis

        U-Net: Convolutional Networks for Biomedical Image Segmentation.
        Ronneberger, O., Fischer, P., & Brox, T. (2015). arXiv preprint arXiv:1505.04597.

    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes the PRLU-Net model.
        Args:
            n_channels (int): Number of input channels (e.g., 1 for grayscale images).
            n_classes (int): Number of output classes for segmentation.
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv1 = self.build_double_conv(n_channels, 64)
        self.in_conv2 = self.build_double_conv(n_channels, 64)

        self.down11 = self.build_down_conv(64, 128)
        self.down12 = self.build_down_conv(128, 256)
        self.down13 = self.build_down_conv(256, 512)
        self.down14 = self.build_down_conv(512, 1024)

        self.down21 = self.build_down_conv(64, 128)
        self.down22 = self.build_down_conv(128, 256)
        self.down23 = self.build_down_conv(256, 512)
        self.down24 = self.build_down_conv(512, 1024)

        self.fuse = nn.Conv2d(1024 * 2, 1024, kernel_size=1)


        self.up21 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv21 = self.build_double_conv(1024 + 512, 512)
        self.up22 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv22 = self.build_double_conv(512 + 256, 256)
        self.up23 = nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2)
        self.conv23 = self.build_double_conv(256 + 128, 128)
        self.up24 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv24 = self.build_double_conv(128 + 64, 64)


        self.head = nn.Conv2d(64, n_classes, kernel_size=1)


    def pad(self, x1, x2):
        """
        Pads the input tensor x1 to match the spatial dimensions of x2.
        Args:
            x1 (torch.Tensor): The input tensor to be padded.
            x2 (torch.Tensor): The tensor whose dimensions x1 should match.
        Returns:
            torch.Tensor: The padded tensor.
        """
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        return F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

    def build_down_conv(self, in_channels, out_channels):
        """
        Builds a downsampling convolutional block with max pooling and double convolution.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.build_double_conv(in_channels, out_channels)
        )

    def build_double_conv(self, in_channels, out_channels):
        """
        Builds a double convolutional block with two convolutional layers, batch normalization, and ReLU activation.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    


    def forward(self, mag, phase):

        """
        Forward pass of the PRLU-Net model.
        Args:
            mag (torch.Tensor): The SWI magnitude input tensor.
            phase (torch.Tensor): The T2* phase input tensor.
        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """

        d11 = self.in_conv1(mag)
        d12 = self.down11(d11)
        d13 = self.down12(d12)
        d14 = self.down13(d13)
        d15 = self.down14(d14)

        d21 = self.in_conv2(phase)
        d22 = self.down21(d21) 
        d23 = self.down22(d22)
        d24 = self.down23(d23)
        d25 = self.down24(d24)


        #Adding spatial connection between the two branches
        fuse = torch.cat([d25, d15], dim=1)
        fuse = self.fuse(fuse)

        u21 = self.up21(fuse)
        u21 = self.pad(u21, d24)    
        x2 = torch.cat([d24, d14, u21], dim=1)
        x2 = self.conv21(x2)

        u22 = self.up22(x2)
        u22 = self.pad(u22, d23)
        x2 = torch.cat([d23, d13, u22], dim=1)
        x2 = self.conv22(x2)

        u23 = self.up23(x2)
        u23 = self.pad(u23, d22)
        x2 = torch.cat([d22, d12, u23], dim=1)
        x2 = self.conv23(x2)

        u24 = self.up24(x2)
        u24 = self.pad(u24, d21)
        x2 = torch.cat([d21, d11, u24], dim=1)
        x2 = self.conv24(x2)

        out2 = self.head(x2)
        return out2


        


    
        
