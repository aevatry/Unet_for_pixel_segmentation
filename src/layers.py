import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)

class DConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super(DConv, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self,x):
        x = self.pool(x)
        return self.conv(x)
    

class UConv(nn.Module):
    def __init__(self, in_channels:int, mid_channels:int, out_channels:int=None):
        super(UConv, self).__init__()

        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(),
        )

        if out_channels!=None:
            self.up_conv = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x  = self.conv(x)
        if self.out_channels != None:
            x = self.up_conv(x)
        return x
