import torch.nn as nn

class C3D(nn.Module):
    """
    This network acts on y * y * 3 warped images and depth maps
    Output is a grid of residuals to modify the warped images input
    """
    def __init__(self, inchannels, outchannels):
        super(type(self), self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        mid = 128
        activation = nn.ELU(inplace=True)
        self.first = ThreexLayer(inchannels, mid, activation)
        self.layer1 = ThreexLayer(mid, mid, activation)
        self.layer2 = ThreexLayer(mid, mid, activation)
        #This activation constrains the values in the last layer
        constrained_activ = nn.Tanh()
        self.final = ThreexLayerNoBN(mid, outchannels, constrained_activ)

        self.weights_init()

    def forward(self, x):
        out = self.first(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.final(out)
        return out

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                print("Initialising ", m, " with Xavier initialisation")
                nn.init.xavier_normal_(m.weight.data)

class ThreexLayer(nn.Module):
    """
    This Layer represents a 3d convolution followed by a batch norm and activation
    With 3 x 3 x 3 filters on a certain number of input and output channels
    """

    def __init__(self, inchannels, outchannels, activation):
        super(type(self), self).__init__()
        self.conv = nn.Conv3d(inchannels, outchannels,
                              kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(outchannels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ThreexLayerNoBN(nn.Module):
    """
    This Layer represents a 3d convolution followed by a batch norm and activation
    With 3 x 3 x 3 filters on a certain number of input and output channels
    """

    def __init__(self, inchannels, outchannels, activation):
        super(type(self), self).__init__()
        self.conv = nn.Conv3d(inchannels, outchannels,
                              kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
