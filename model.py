import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim

def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.5, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=0)
        self.c8=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)
        self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)
        self.l_c1=nn.Linear(128,n_outputs)
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(256)
        self.bn7=nn.BatchNorm2d(512)
        self.bn8=nn.BatchNorm2d(256)
        self.bn9=nn.BatchNorm2d(128)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c7(h)
        h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h=self.c8(h)
        h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h=self.c9(h)
        h=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit=self.l_c1(h)
        if self.top_bn:
            logit=call_bn(self.bn_c1, logit)
        return logit

import torch.nn as nn
import math


def _conv2d_bn(in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    bn = nn.BatchNorm2d(num_features=out_channels)
    return nn.Sequential(conv, bn)


def _conv2d_bn_relu(in_channels, out_channels, kernel_size, stride, padding):
    conv2d_bn = _conv2d_bn(in_channels, out_channels, kernel_size, stride, padding)
    relu = nn.ReLU(inplace=True)
    layers = list(conv2d_bn.children())
    layers.append(relu)
    return nn.Sequential(*layers)


class _BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downscale=False):
        super(_BasicBlock, self).__init__()
        self.down_sampler = None
        stride = 1
        if downscale:
            self.down_sampler = _conv2d_bn(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
            stride = 2
        self.conv_bn_relu1 = _conv2d_bn_relu(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # don't relu here! relu on (H(x) + x)
        self.conv_bn2 = _conv2d_bn(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu_out = nn.ReLU(inplace=True)
        # residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        # residual = nn.BatchNorm2d(num_features=out_channels)
        # residual = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        if self.down_sampler:
            input = self.down_sampler(x)
        residual = self.conv_bn_relu1(x)
        residual = self.conv_bn2(residual)
        out = self.relu_out(input + residual)
        return out


class _ResNet(nn.Module):
    def __init__(self, num_layer_stack):
        super(_ResNet, self).__init__()
        self.conv1 = _conv2d_bn_relu(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.layer1 = self.__make_layers(num_layer_stack, in_channels=16, out_channels=16, downscale=False)
        self.layer2 = self.__make_layers(num_layer_stack, in_channels=16, out_channels=32, downscale=True)
        self.layer3 = self.__make_layers(num_layer_stack, in_channels=32, out_channels=64, downscale=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(in_features=64, out_features=100)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def __make_layers(self, num_layer_stack, in_channels, out_channels, downscale):
        layers = []
        layers.append(_BasicBlock(in_channels=in_channels, out_channels=out_channels, downscale=downscale))
        for i in range(num_layer_stack - 1):
            layers.append(_BasicBlock(in_channels=out_channels, out_channels=out_channels, downscale=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


def resnet20():
    return _ResNet(num_layer_stack=3)


def resnet32():
    return _ResNet(num_layer_stack=5)


def resnet56():
    return _ResNet(num_layer_stack=9)