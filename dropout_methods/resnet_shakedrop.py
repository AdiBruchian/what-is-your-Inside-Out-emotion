import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ShakeDropFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[0, 1], beta_range=[0, 1]):
        """
        Forward pass for ShakeDrop function.
        
        Args:
            x: Input tensor.
            training: If True, the model is in training mode.
            p_drop: Probability of dropping.
            alpha_range: Range for alpha.
            beta_range: Range for beta.
            
        Returns:
            Output tensor after applying ShakeDrop.
        """
        ctx.save_for_backward(x)
        ctx.p_drop = p_drop
        ctx.alpha_range = alpha_range
        ctx.beta_range = beta_range

        if training:
            gate = torch.cuda.FloatTensor([0]).bernoulli_(1 - p_drop)
            ctx.save_for_backward(gate)

            if gate.item() == 0:
                alpha = torch.cuda.FloatTensor(x.size(0)).uniform_(*alpha_range)
                alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                alpha = Variable(torch.zeros(1)).cuda()
                return alpha * x
            else:
                return x
        else:
            return (1 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for ShakeDrop function.
        
        Args:
            grad_output: Gradient of the loss with respect to the output of this layer.
            
        Returns:
            Gradient of the loss with respect to the input of this layer.
        """
        gate = ctx.saved_tensors[0]
        p_drop = ctx.p_drop
        beta_range = ctx.beta_range
        
        if gate.item() == 0:
            beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_(*beta_range)
            beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
            beta = Variable(beta)
            return beta * grad_output, None, None, None, None, None
        else:
            return grad_output, None, None, None, None, None

class ShakeDrop(nn.Module):

    def __init__(self, p_drop=0.5, alpha_range=[0, 1], beta_range=[0, 1]):
        """
        ShakeDrop module initializer.
        
        Args:
            p_drop: Probability of dropping.
            alpha_range: Range for alpha.
            beta_range: Range for beta.
        """
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range
        self.beta_range = beta_range

    def forward(self, x):
        """
        Forward pass for ShakeDrop module.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after applying ShakeDrop.
        """
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range, self.beta_range)


def Conv1(in_planes, places, stride=2):
    """
    1x1 Convolutional layer with BatchNorm and ReLU activation.
    
    Args:
        in_planes: Number of input channels.
        places: Number of output channels.
        stride: Stride of the convolution.
        
    Returns:
        A convolutional block.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p_shakedrop=1.0):
        """
        Basic residual block used in ResNet.
        
        Args:
            in_planes: Number of input channels.
            planes: Number of output channels.
            stride: Stride of the convolution.
            p_shakedrop: ShakeDrop probability.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shake_drop = ShakeDrop(p_shakedrop)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        """
        Forward pass for BasicBlock.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after applying the block.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shake_drop(out)
        if out.mean() == 0:
            out = self.shortcut(x)
        else:
            out = out + self.shortcut(x)
            out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7, alpha_range=[-1, 1], beta_range=[0, 1]):
        """
        ResNet initializer.
        
        Args:
            block: Type of block to use.
            num_blocks: Number of blocks in each layer.
            num_classes: Number of output classes.
            alpha_range: Range for alpha.
            beta_range: Range for beta.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv1(in_planes=3, places=64)

        n = sum(num_blocks)
        self.ps_shakedrop = [0.5 / n * (i + 1) for i in range(n)]

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Create a layer of blocks.
        
        Args:
            block: Type of block to use.
            planes: Number of output channels.
            num_blocks: Number of blocks in the layer.
            stride: Stride of the convolution.
            
        Returns:
            A sequential block layer.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, p_shakedrop=self.ps_shakedrop.pop(0)))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for ResNet.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after applying ResNet.
        """
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet18_shakedrop(dropout_rate, num_classes=7):
    """
    Function to create a ResNet-18 model with ShakeDrop regularization.
    
    Args:
        alpha_range: Range for alpha.
        beta_range: Range for beta.
        num_classes: Number of output classes.
        
    Returns:
        A ResNet-18 model with ShakeDrop regularization.
    """
    alpha_range, beta_range = dropout_rate
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, alpha_range=alpha_range, beta_range=beta_range)
