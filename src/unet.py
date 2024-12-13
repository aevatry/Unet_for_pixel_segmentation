import torch
import torch.nn as nn
import numpy as np

from layers import DConv, UConv, DoubleConv

class Unet(nn.Module):
    def __init__(self, in_channels:int = 3, num_classes:int = 1):
        super(Unet, self).__init__()

        self.up_layers = nn.ModuleList([])
        self.down_layers = nn.ModuleList([])

        base_shapes =[ 64, 128, 256, 512]
        up_shapes = [base_shapes[-1]*2] + base_shapes[::-1] + [None]
        
        self.first_layer = DoubleConv(in_channels=in_channels, out_channels=base_shapes[0])

        # down convolutions
        for i in range(len(base_shapes)-1):
            self.down_layers.append(DConv(in_channels=base_shapes[i], out_channels=base_shapes[i+1]))

        # mid convolution at transition
        self.mid_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=base_shapes[-1], out_channels=up_shapes[0], kernel_size=3),
            nn.BatchNorm2d(num_features=up_shapes[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=up_shapes[0], out_channels=up_shapes[0], kernel_size=3),
            nn.BatchNorm2d(num_features=up_shapes[0]),
            nn.ReLU(), 
            nn.ConvTranspose2d(in_channels=up_shapes[0], out_channels=up_shapes[1], kernel_size=2, stride=2)           
        )

        # up convolutions
        for i in range(len(up_shapes)-2):
            self.up_layers.append(UConv(in_channels=up_shapes[i], mid_channels=up_shapes[i+1], out_channels=up_shapes[i+2]))

        # convolution for outputs
        self.out_layer = nn.Conv2d(in_channels=up_shapes[-2], out_channels=num_classes, kernel_size=3)


    def forward(self, x:torch.Tensor):

        x = self.first_layer(x)
        down_outputs = [x]
        for layer in self.down_layers:
            x = layer(x)
            down_outputs+=[x]
            print(x.shape)

        x = self.mid_layer(x)
        print("output of mid layer", x.shape)

        for i, layer in enumerate(self.up_layers):
            x_prev = self.crop_input_to_upconv(down_outputs[-(i+1)], x.shape)
            x = torch.cat((x_prev, x), dim=1)
            print(x.shape)
            x = layer(x)

        x = self.out_layer(x) 

        return x
    
    def crop_input_to_upconv(self, out_conv_down:torch.Tensor, wanted_shape:torch.Size)->torch.Tensor:
        size_to_crop = np.array(out_conv_down.shape[2:]) - np.array(wanted_shape[2:])
        size_to_crop = size_to_crop/2

        # get the actual values we need to crop by
        s_up_left = np.floor(size_to_crop).astype(int)
        s_down_right = np.ceil(size_to_crop).astype(int)


        out = out_conv_down[:, :, s_up_left[0]:-s_down_right[0], s_up_left[1]:-s_down_right[1]]
        return out

        

