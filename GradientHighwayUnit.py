__author__ = 'Titi'

import torch
import torch.nn as nn

class GHU(nn.Module):
    def __init__(self, in_channel, num_hidden, h, w, filter_size, stride):
        """
        :param in_channel: input tensor channel
        :param num_hidden: output tensor channel
        :param size: input tensor size
        :param filter_size: The filter size of convolution in the lstm
        :param stride: The stride of convolution in the lstm
        """
        
        super(GHU, self).__init__()
        
        self.num_hidden = num_hidden
        padding = filter_size // 2
        
        self.conv_z = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=padding),
            nn.LayerNorm([num_hidden * 2, h, w])
        )
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=padding),
            nn.LayerNorm([num_hidden * 2, h, w])
        )
        
    def forward(self, x, z):
        z_concat = self.conv_z(z)
        x_concat = self.conv_x(x)
        
        gates = torch.add(x_concat, z_concat)
        p, u = torch.split(gates, self.num_hidden, dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1-u) * z
        return z_new
