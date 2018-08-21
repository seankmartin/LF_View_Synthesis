import torch.nn as nn
import common

class DepthModel(nn.Module):
    def __init__(self, inchannels, outchannels, max_disp=20):
        super(DepthModel, self).__init__()
        kernel_size = 3
        self.max_disp=max_disp
        self.act = nn.ELU(inplace=True)
        self.conv1 = common.default_conv(inchannels, 16, kernel_size)
        self.conv2 = common.default_conv(16, 64, kernel_size)
        # A dilated convolution has output size:
        # (size + 2p - k_size - (k_size - 1)(d - 1)) // 2 + 1
        self.d_conv1 = nn.Conv2d(64, 128, 3, padding=2, dilation=2)
        self.d_conv2 = nn.Conv2d(128, 128, 3, padding=4, dilation=4)
        self.d_conv3 = nn.Conv2d(128, 128, 3, padding=8, dilation=8)
        self.d_conv4 = nn.Conv2d(128, 128, 3, padding=16, dilation=16)

        self.conv3 = common.default_conv(128, 128, kernel_size)
        self.conv4 = common.default_conv(128, 64, kernel_size)
        self.last_act = nn.Tanh()
    
    def forward(self, x):
        res = x
        res = self.conv1(res)
        res = self.act(res)
        res = self.conv2(res)
        res = self.act(res)
        res = self.d_conv1(res)
        res = self.act(res)
        res = self.d_conv2(res)
        res = self.act(res)
        res = self.d_conv3(res)
        res = self.act(res)
        res = self.d_conv4(res)
        res = self.act(res)
        res = self.conv3(res)
        res = self.act(res)
        res = self.conv4(res)
        res = self.last_act(res)
        res = res * self.max_disp
        return res



