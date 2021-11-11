
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU
from torch.nn import ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d
from torch.nn import AdaptiveAvgPool2d, Sequential, Module, Parameter
from torch import nn
import torch
import math
from .model_utils import Flatten, l2_norm

class ConvBlock(Module):
    def __init__(self, 
                 in_c, out_c, 
                 kernel = (1, 1), 
                 stride = (1, 1), 
                 padding = (0, 0), 
                 groups = 1):
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_c,
                           out_channels = out_c, 
                           kernel_size = kernel, 
                           groups = groups, 
                           stride = stride, 
                           padding = padding, 
                           bias = False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, 
                 in_c, out_c, 
                 kernel = (1, 1), 
                 stride = (1, 1), 
                 padding = (0, 0), 
                 groups = 1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c,
                           out_channels = out_c, 
                           kernel_size = kernel, 
                           groups = groups, 
                           stride = stride, 
                           padding = padding, 
                           bias = False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class DepthWise(Module):
    def __init__(self, 
                 in_c, out_c, 
                 residual = False, 
                 kernel = (3, 3), 
                 stride = (2, 2), 
                 padding = (1, 1), 
                 groups = 1):
        super(DepthWise, self).__init__()
        self.conv = ConvBlock(in_c, 
                              out_c = groups, 
                              kernel = (1, 1), 
                              padding = (0, 0), 
                              stride = (1, 1))
        self.conv_dw = ConvBlock(groups, 
                                 groups, 
                                 groups = groups, 
                                 kernel = kernel, 
                                 padding = padding, 
                                 stride = stride)
        self.project = Linear_block(groups, 
                                    out_c, 
                                    kernel = (1, 1), 
                                    padding = (0, 0), 
                                    stride = (1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, 
                 c, num_block, 
                 groups, 
                 kernel = (3, 3), 
                 stride = (1, 1), 
                 padding = (1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(DepthWise(c, c, 
                                     residual = True, 
                                     kernel = kernel, 
                                     padding = padding, 
                                     stride = stride, 
                                     groups = groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)



class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(3, 64,
                                kernel = (3, 3), 
                                stride = (2, 2), 
                                padding = (1, 1))
        self.conv2_dw = ConvBlock(64, 64, 
                                   kernel = (3, 3), 
                                   stride = (1, 1),
                                   padding = (1, 1),
                                   groups = 64)
        self.conv_23 = DepthWise(64, 64, 
                                  kernel = (3, 3), 
                                  stride = (2, 2), 
                                  padding = (1, 1), 
                                  groups = 128)
        self.conv_3 = Residual(64, 
                               num_block = 4, 
                               groups = 128, 
                               kernel = (3, 3), 
                               stride = (1, 1), 
                               padding = (1, 1))
        self.conv_34 = DepthWise(64, 128, 
                                  kernel = (3, 3), 
                                  stride = (2, 2), 
                                  padding = (1, 1), 
                                  groups = 256)
        self.conv_4 = Residual(128, 
                               num_block = 6, 
                               groups = 256, 
                               kernel = (3, 3), 
                               stride = (1, 1), 
                               padding = (1, 1))
        self.conv_45 = DepthWise(128, 128, 
                                  kernel = (3, 3), 
                                  stride = (2, 2), 
                                  padding = (1, 1),
                                  groups = 512)
        self.conv_5 = Residual(128, 
                               num_block = 2, 
                               groups = 256, 
                               kernel = (3, 3), 
                               stride = (1, 1), 
                               padding = (1, 1))
        self.conv_6_sep = ConvBlock(128, 512, 
                                     kernel = (1, 1), 
                                     stride = (1, 1), 
                                     padding = (0, 0))
        self.conv_6_dw = Linear_block(512, 512,
                                      groups = 512, 
                                      kernel = (7,7), 
                                      stride = (1, 1), 
                                      padding = (0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias = False)
        self.bn = BatchNorm1d(embedding_size)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)     
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)



if __name__ == "__main__":

    tensor_input = torch.Tensor(2, 3, 112, 112)
    net = MobileFaceNet(512)
    x = net(tensor_input)
    print(x.shape)



