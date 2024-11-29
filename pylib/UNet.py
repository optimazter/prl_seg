import torch
from torch import nn

#https://debuggercafe.com/unet-from-scratch-using-pytorch/

NUM_CLASSES = 3
IN_CHANNELS = 1




class PRLSegUNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.contractions = self.build_contraction([IN_CHANNELS, 64, 128, 256, 612, 1024])
        self.expansions = self.build_expansion([1024, 512, 256, 128, 64])
        self.head = nn.Conv2d(in_channels=64, out_channels=NUM_CLASSES)

    def forward(self, x):
        out_contracted = []
        out = x
        for i, contraction in enumerate(self.contractions):
            out = contraction(out)
            if i != len(self.contractions):
                out_contracted.append(out)
                out = self.max_pool(out)

        for j, (exp, T) in enumerate(self.expansions):
            out_T = T(out)
            out = exp(torch.cat([out_contracted[-j], out_T], 1))
        
        out = self.head(out)
        return out


    def build_expansion(self, channels: list) -> zip:
        exp = []
        T = []
        for i in range(len(channels) - 1):
            T.append(nn.ConvTranspose2d(channels[i], channels[i + 1]))
            exp.extend(self.build_double_conv_2d(channels[i], channels[i + 1]))
        return zip(exp, T)

    def build_contraction(self, channels: list) -> list:
        con = []
        N = len(channels) - 1
        for i in range(N):
            con.extend(self.build_double_conv_2d(channels[i], channels[i+1]))
        return con
    
    
    def build_double_conv_2d(self, in_channels, out_channels) -> list:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        ]

        
