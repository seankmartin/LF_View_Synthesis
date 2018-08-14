import torch.nn as nn
import pdb
import math

#3x3 convolution with padding and stride of 1:
def conv3_3(inplanes, outplanes):
    return nn.Conv2d(inplanes, outplanes, kernel_size = 3, stride = 1, padding = 1, bias = False)

def conv5_5(inplanes, outplanes):
    return nn.Conv2d(inplanes, outplanes, kernel_size = 5, stride = 1, padding = 2, bias = False)

#When the size of the depth changes in the network, the residual also needs to be rescaled
def shortcut(inplanes, outplanes):
    sample = nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size = 1, stride = 1, bias = False),
        nn.BatchNorm2d(outplanes)
    )
    return sample

def convpool(inplanes, outplanes, activation):
    sample = nn.Sequential(
        nn.ZeroPad2d(2),
        nn.ZeroPad2d((1, 0, 1, 0)),
        nn.Conv2d(inplanes, outplanes, kernel_size = 3, stride = 2, bias = False),
        nn.BatchNorm2d(outplanes),
        activation
    )
    return sample

#REFERENCE see Striving for Simplicity - the all convolutional net
#The purpose of this class is to replace pooling by a convolution
class ConvPool(nn.Module):
    def __init__(self, inplanes, outplanes, activation, kernel_size = 3, stride = 2):
        super(ConvPool, self).__init__()
        #Zero pad the left and top by 1
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, bias = False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.activation = activation

    def forward(self, x):
         out = self.pad(x)
         out = self.conv(out)
         out = self.bn(out)
         out = self.activation(out)
         return out

#This class handles the basic structure of our CNN
#Ripool will be built from layers of these blocks
#This idea is from Resnet and GoogleNet
class Block(nn.Module):
    def __init__(self, inplanes, outplanes, activation, downsample = False):
        super(Block, self).__init__()
        self.conv_pooling = ConvPool(inplanes, outplanes, activation)
        #self.conv_pool = convpool(inplanes, outplanes, activation)
        self.conv1 = conv3_3(inplanes, outplanes)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.activation = activation
        self.conv2 = conv3_3(outplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.shortcut = shortcut(inplanes, outplanes)
        self.downsample = downsample
    
    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if(self.downsample):
            residual = self.shortcut(x)
        else:
            residual = x

        out += residual
        out = self.activation(out)
        return out

class PoolBlock(nn.Module):  
    def __init__(self, inplanes, outplanes, activation, downsample = False):
        super(PoolBlock, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pooling = nn.MaxPool2d(kernel_size = 3, stride = 2)
        #self.conv_pool = convpool(inplanes, outplanes, activation)
        self.conv1 = conv3_3(inplanes, outplanes)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.activation = activation
        self.conv2 = conv3_3(outplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.shortcut = shortcut(inplanes, outplanes)
        self.downsample = downsample
    
    def forward(self, x):
        out = x
        if(self.downsample):
            out = self.pad(out)
            out = self.pooling(out)
       
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if(self.downsample):
            residual = self.shortcut(x)
        else:
            residual = x

        out += residual
        out = self.activation(out)
        return out      

#The same as above except it uses 5x5 filters instead of 3x3
class Block5(nn.Module):
    def __init__(self, inplanes, outplanes, activation, downsample = False):
        super(Block5, self).__init__()
        self.conv_pooling = ConvPool(inplanes, outplanes, activation)
        #self.conv_pool = convpool(inplanes, outplanes, activation)
        self.conv1 = conv5_5(inplanes, outplanes)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.activation = activation
        self.conv2 = conv5_5(outplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.shortcut = shortcut(inplanes, outplanes)
        self.downsample = downsample
    
    def forward(self, x):
        out = x
        if(self.downsample):
            out = self.conv_pooling(out)
        else:
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if(self.downsample):
            residual = self.shortcut(x)
        else:
            residual = x

        out += residual
        out = self.activation(out)
        return out

#REFERENCE the structure of this class is very similar to the structure of resnet at torchvision
#See https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class BigRipool(nn.Module):
    def __init__(
            self, layers, activation_fn, thin, 
            inchannels, blockType=Block, num_classes=192
        ):
        super(BigRipool, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 64, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        if(activation_fn == nn.ReLU):
            self.activation = activation_fn(inplace = True)
        if(activation_fn == nn.ELU):
            self.activation = activation_fn(inplace = True)
        else :
            self.activation = activation_fn()
        if(not thin):
            self.layer1 = self.make_layer(blockType, 64, 64, layers[0], False)
            self.layer2 = self.make_layer(blockType, 64, 128, layers[1], True)
            self.layer3 = self.make_layer(blockType, 128, 256, layers[2], True)
            self.layer4 = self.make_layer(blockType, 256, num_classes, layers[3], True)
            self.layer5 = self.make_layer_tanh(blockType, num_classes, num_classes, 1, False)
        else:
            self.layer1 = self.make_layer(blockType, 64, 32, layers[0], True)
            self.layer2 = self.make_layer(blockType, 32, 64, layers[1], True)
            self.layer3 = self.make_layer(blockType, 64, 128, layers[2], True)
            self.layer4 = self.make_layer(blockType, 128, num_classes, layers[3], True)
            self.layer5 = self.make_layer_tanh(blockType, num_classes, num_classes, 1, False)


        #Initialise the weights
        for m in self.modules():
            #Based on the normal distribution
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.xavier_normal_(m.weight.data)

    #Creates a layer of blocks 
    def make_layer(self, blockType, inplanes, outplanes, num_blocks, downsample):
        layers = []
        layers.append(blockType(inplanes, outplanes, self.activation, downsample))
        for i in range(1, num_blocks):
            layers.append(blockType(outplanes, outplanes, self.activation, False))
        #unpack the list and pass it to Sequential
        return nn.Sequential(*layers)
    
    def make_layer_tanh(self, blockType, inplanes, outplanes, num_blocks, downsample):
        layers = []
        layers.append(blockType(inplanes, outplanes, nn.Tanh(), downsample))
        for i in range(1, num_blocks):
            layers.append(blockType(outplanes, outplanes, nn.Tanh(), False))
        #unpack the list and pass it to Sequential
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        #Now into the blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class SimpleRipool(nn.Module):
    def __init__(self, activation_fn = nn.ReLU, nClasses = 200):
        super(SimpleRipool, self).__init__()
        if(activation_fn == nn.ReLU):
            self.activation = activation_fn(inplace = True)
        else :
            self.activation = activation_fn()
        self.conv0 = nn.Conv2d(3, 64, kernel_size = 5, padding = 2, stride = 1)
        self.batch_norm0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        #This layer acts like pooling and convolving
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 2, padding = 0, stride = 2)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 2, padding = 0, stride = 2)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(32768, 1024)
        self.batch_norm5 = nn.BatchNorm1d(1024)
        #self.dropout_1 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(1024, nClasses)

        #Initialise the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        y = self.conv0(x)
        y = self.batch_norm0(y)
        y = self.activation(y)

        residual = y
        y = self.conv1(y)
        y = self.batch_norm1(y)
        y = self.activation(y)

        y = self.conv2(y)
        y = self.batch_norm2(y)
        y += residual
        y = self.activation(y)

        y = self.conv3(y)
        y = self.batch_norm3(y)
        y = self.activation(y)

        y = self.conv4(y)
        y = self.batch_norm4(y)
        y = self.activation(y)

        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.batch_norm5(y)
        y = self.activation(y)
        y = self.dropout_1(y)

        y = self.fc2(y)
        return y