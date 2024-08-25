import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import math

class VariationalDropout(nn.Module):
    def __init__(self, input_size, out_size, log_sigma2=-10, threshold=3):
        """
        :param input_size: An int of input size
        :param log_sigma2: Initial value of log sigma ^ 2.
               It is crusial for training since it determines initial value of alpha
        :param threshold: Value for thresholding of validation. If log_alpha > threshold, then weight is zeroed
        :param out_size: An int of output size
        """
        super(VariationalDropout, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.theta = Parameter(torch.FloatTensor(input_size, out_size))
        self.bias = Parameter(torch.Tensor(out_size))

        self.log_sigma2 = Parameter(torch.FloatTensor(input_size, out_size).fill_(log_sigma2))

        self.reset_parameters()

        self.k = [0.63576, 1.87320, 1.48695]

        self.threshold = threshold

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)

        self.theta.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def clip(input, to=8):
        input = input.masked_fill(input < -to, -to)
        input = input.masked_fill(input > to, to)

        return input

    def kld(self, log_alpha):

        first_term = self.k[0] * F.sigmoid(self.k[1] + self.k[2] * log_alpha)
        second_term = 0.5 * torch.log(1 + torch.exp(-log_alpha))

        return -(first_term - second_term - self.k[0]).sum() / (self.input_size * self.out_size)

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :return: An float tensor with shape of [batch_size, out_size] and negative layer-kld estimation
        """

        log_alpha = self.clip(self.log_sigma2 - torch.log(self.theta ** 2))
        kld = self.kld(log_alpha)

        if not self.training:
            mask = log_alpha > self.threshold
            return torch.addmm(self.bias, input, self.theta.masked_fill(mask, 0))

        mu = torch.mm(input, self.theta)
        std = torch.sqrt(torch.mm(input ** 2, self.log_sigma2.exp()) + 1e-6)

        eps = Variable(torch.randn(*mu.size()))
        if input.is_cuda:
            eps = eps.cuda()

        return std * eps + mu + self.bias, kld

    def max_alpha(self):
        log_alpha = self.log_sigma2 - self.theta ** 2
        return torch.max(log_alpha.exp())
    



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




class ResNet_with_Variational(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet_with_Variational, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.variational_dropout = VariationalDropout(512 * block.expansion, 512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        kld_loss = 0
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.variational_dropout(x)
        if isinstance(x, tuple):
            x, kld = x
            kld_loss += kld
        x = self.fc(x)
        return x, kld_loss

def ResNet18_Variational(dropout_rate = 0.0, num_classes=7):
    return ResNet_with_Variational(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

