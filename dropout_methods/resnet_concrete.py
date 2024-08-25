import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np



class ConcreteDropout(nn.Module):
    """Concrete Dropout.
    Implementation of the Concrete Dropout module as described in the
    'Concrete Dropout' paper: https://arxiv.org/pdf/1705.07832
    """

    def __init__(self,
                 weight_regulariser: float,
                 dropout_regulariser: float,
                 init_min: float = 0.1,
                 init_max: float = 0.1) -> None:

        """Concrete Dropout.

        Parameters
        ----------
        weight_regulariser : float
            Weight regulariser term.
        dropout_regulariser : float
            Dropout regulariser term.
        init_min : float
            Initial min value.
        init_max : float
            Initial max value.
        """

        super().__init__()

        self.weight_regulariser = weight_regulariser
        self.dropout_regulariser = dropout_regulariser

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)

        self.regularisation = 0.0

    def forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:

        """Calculates the forward pass.

        The regularisation term for the layer is calculated and assigned to a
        class attribute - this can later be accessed to evaluate the loss.

        Parameters
        ----------
        x : Tensor
            Input to the Concrete Dropout.
        layer : nn.Module
            Layer for which to calculate the Concrete Dropout.

        Returns
        -------
        Tensor
            Output from the dropout layer.
        """

        output = layer(self._concrete_dropout(x))

        sum_of_squares = 0
        for param in layer.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        weights_reg = self.weight_regulariser * sum_of_squares / (1.0 - self.p)

        dropout_reg = self.p * torch.log(self.p)
        dropout_reg += (1.0 - self.p) * torch.log(1.0 - self.p)
        dropout_reg *= self.dropout_regulariser * x[0].numel()

        self.regularisation = weights_reg + dropout_reg

        return output

    def _concrete_dropout(self, x: torch.Tensor) -> torch.Tensor:

        """Computes the Concrete Dropout.

        Parameters
        ----------
        x : Tensor
            Input tensor to the Concrete Dropout layer.

        Returns
        -------
        Tensor
            Outputs from Concrete Dropout.
        """

        eps = 1e-7
        tmp = 0.1

        self.p = torch.sigmoid(self.p_logit)
        u_noise = torch.rand_like(x)

        drop_prob = (torch.log(self.p + eps) -
                     torch.log(1 - self.p + eps) +
                     torch.log(u_noise + eps) -
                     torch.log(1 - u_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / tmp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x = torch.mul(x, random_tensor) / retain_prob

        return x
    


class BasicBlock(nn.Module):
    """
    Basic building block for ResNet.
    
    Each BasicBlock consists of two convolutional layers with BatchNorm and ReLU activation.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        Initialize BasicBlock.

        Parameters
        ----------
        in_planes : int
            Number of input channels.
        planes : int
            Number of output channels.
        stride : int
            Stride of the convolutional layers.
        """
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
        """
        Forward pass through the BasicBlock.

        Parameters
        ----------
        x : Tensor
            Input tensor to the block.

        Returns
        -------
        Tensor
            Output tensor after passing through the block.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_with_Concrete(nn.Module):
    """
    ResNet with Concrete Dropout integration.
    
    This model implements ResNet architecture and uses ConcreteDropout in the final fully connected layer.
    """

    def __init__(self, block, num_blocks, num_classes=7):
        """
        Initialize ResNet with Concrete Dropout.

        Parameters
        ----------
        block : nn.Module
            The block type to use (e.g., BasicBlock).
        num_blocks : list
            Number of blocks in each layer of the ResNet.
        num_classes : int
            Number of output classes.
        """
        super(ResNet_with_Concrete, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.concrete_dropout_fc = ConcreteDropout(weight_regulariser=1e-4, dropout_regulariser=1e-4, init_min=0.1, init_max=0.1)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Create a sequential layer of ResNet blocks.

        Parameters
        ----------
        block : nn.Module
            The block type to use.
        planes : int
            Number of output channels for the blocks.
        num_blocks : int
            Number of blocks in the layer.
        stride : int
            Stride of the first convolutional block in the layer.

        Returns
        -------
        nn.Sequential
            Sequential container of ResNet blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the ResNet model.

        Parameters
        ----------
        x : Tensor
            Input tensor to the model.

        Returns
        -------
        Tensor
            Output tensor after passing through the model.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.concrete_dropout_fc(x, self.fc)

        return x


def ResNet18_Concrete(dropout_rate=0.0, num_classes=7):
    """
    Create a ResNet-18 model with Concrete Dropout.

    Parameters
    ----------
    dropout_rate : float
        Dropout rate to be used in Concrete Dropout.
    num_classes : int
        Number of output classes.

    Returns
    -------
    ResNet_with_Concrete
        A ResNet-18 model with Concrete Dropout.
    """
    return ResNet_with_Concrete(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
