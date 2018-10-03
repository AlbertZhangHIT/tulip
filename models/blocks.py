"""ResNet blocks, aka an Euler step."""
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import math

from .shake import shake


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels,  stride=1,
            kernel_size=(3,3), dropout=0., bn=False,
            **kwargs):
        """A basic 2d ResNet block, with modifications on original ResNet paper
        [1].  Dropout (if active) is placed between convolutions, which is the
        convention adopted in [2]. Every convolution is followed by batch
        normalization (if active).

        When the number of input channels differs from output channels, a fully
        connected linear layer on the channels is performed first, to
        grow/shrink input channels to the output shape.

        Similarly, if stride>1, then a spatial averaging kernel is performed
        last, to reduce the spatial dimensions.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride (int, optional): stride of the convolutions (default: 1)
            kernel_size (tuple, optional): kernel shape (default: 3)
            dropout (float, optional): dropout probability (default: 0)
            bn (bool, optional): turn on batch norm (default: False)

        [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2016.
            Deep Residual Learning for Image Recognition. arXiv:1512.03385
        [2] Zagoruyko, S and Komodakis, N, 2016. Wide residual networks.
            arXiv:1605.07146.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = _pair(kernel_size)

        padding = [k //2 for k in kernel_size]

        self.fcc = None
        if not out_channels==in_channels:
            fcc = [nn.Conv2d(in_channels, out_channels, 1, bias = not bn, stride=1)]
            w = fcc[0].weight.data
            w = w.view(out_channels, -1)
            w = F.softmax(w,-1)
            fcc[0].weight.data.copy_(w.view(out_channels, in_channels, 1,1))
            if bn:
                fcc.append(nn.BatchNorm2d(out_channels, affine=True))
            self.fcc = nn.Sequential(*fcc)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        self.convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels,
                kernel_size, groups=1, 
                bias=not bn, padding=padding, stride=1) for l in range(2)])


        if bn:
            self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels,
                                          affine=True),
                                      nn.BatchNorm2d(out_channels,
                                          affine=True)])
        else:
            self.bns = None


        self.sigma = nn.ReLU()
        self.nonlinear = True

        self.avg = nn.AvgPool2d(stride) if stride>1 else None



    def forward(self, x):
        if self.fcc is not None:
            x = self.fcc(x)
            if self.nonlinear:
                x = self.sigma(x)

        y = self.convs[0](x)

        if self.bns is not None:
            y = self.bns[0](y)

        if self.nonlinear:
            y = self.sigma(y)

        if self.dropout is not None:
            y = self.dropout(y)

        y = self.convs[1](y)

        if self.bns is not None:
            y = self.bns[1](y)

        if self.nonlinear:
            y = self.sigma(x+y)

        if self.avg is not None:
            y = self.avg(y)

        return y


class BranchBlock(nn.Module):

    def __init__(self, in_channels, out_channels,  stride=1,
            kernel_size=(3,3), dropout=0., bn=False, branches=2, method='mean',
            **kwargs):
        """A 2d ResNet block, where the channels are 'branched' into separate
        groups.  Dropout (if active) is placed between convolutions, which is
        the convention adopted in [2]. Every convolution is followed by batch
        normalization (if active).

        When the number of input channels differs from output channels, a fully
        connected linear layer on the channels is performed first, to
        grow/shrink input channels to the output shape.

        Similarly, if stride>1, then a spatial averaging kernel is performed
        last, to reduce the spatial dimensions.

        The branches are aggregated either by taking the mean, max, or min,
        depending on the argument "method". Default is to take the mean.  If
        method='shake', then a mean is taken during model evaluation, but a
        random convex combination is used during training.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride (int, optional): stride of the convolutions (default: 1)
            kernel_size (tuple, optional): kernel shape (default: 3)
            dropout (float, optional): dropout probability (default: 0)
            bn (bool, optional): turn on batch norm (default: False)
            branches (int, optional): number of branches (default: 2)
            method (string, optional): aggregation method. One of 'mean',
            'max', 'min' or 'shake'. (default: mean)
            concentration (float, optional): if method='shake', this is the
                concentration hyperparameter of the Dirichlet distribution
                (default:1)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = _pair(kernel_size)
        self.branches = branches
        self.method = method

        if method=='shake':
            try:
                self.concentration = kwargs['concentration']
            except:
                self.concentration = 1.
        else:
            self.concentration = None

        assert method in ['shake', 'mean', 'min', 'max']

        padding = [k //2 for k in kernel_size]

        self.fcc = None
        if not out_channels==in_channels:
            fcc = [nn.Conv2d(in_channels, out_channels, 1, bias = not bn, stride=1)]
            w = fcc[0].weight.data
            w = w.view(out_channels, -1)
            w = F.softmax(w,-1)
            fcc[0].weight.data.copy_(w.view(out_channels, in_channels, 1,1))
            if bn:
                fcc.append(nn.BatchNorm2d(out_channels))
                fcc[1].weight.data.fill_(1.)
            self.fcc = nn.Sequential(*fcc)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        self.conv0 = nn.Conv2d(out_channels, out_channels*branches,
                                kernel_size, groups=1,
                                bias=not bn, padding=padding, stride=1)
        self.conv1 = nn.Conv2d(out_channels*branches, out_channels*branches,
                                kernel_size, groups=branches,
                                bias=not bn, padding=padding, stride=1)

        if bn:
            self.bn0 = nn.BatchNorm2d(out_channels * branches)
            self.bn1 = nn.BatchNorm2d(out_channels * branches)
        else:
            self.bn0, self.bn1 = [None]*2


        self.sigma = nn.ReLU()
        self.nonlinear = True

        self.avg = nn.AvgPool2d(stride) if stride>1 else None


    def forward(self, x):
        if self.fcc is not None:
            x = self.fcc(x)
            if self.nonlinear:
                x = self.sigma(x)

        y = self.conv0(x)

        if self.bn0 is not None:
            y = self.bn0(y)

        if self.nonlinear:
            y = self.sigma(y)

        if self.dropout is not None:
            y = self.dropout(y)

        y = self.conv1(y)

        if self.bn1 is not None:
            y = self.bn1(y)


        y = y.unsqueeze(1).chunk(self.branches,dim=2)
        y = th.cat(y, 1)

        if self.method=='shake':
            if self.nonlinear:
                y = shake(y, 1, 0, self.concentration, self.training)
            else:
                y = y.mean(1)
        elif self.method=='mean':
            y = y.mean(1)
        elif self.method=='max':
            if not self.nonlinear:
                raise ValueError('Linear approximation not possible with aggregation method "max"')
            y = y.max(1)[0]
        elif self.method=='min':
            if not self.nonlinear:
                raise ValueError('Linear approximation not possible with aggregation method "min"')
            y = y.min(1)[0]

        y = x+y
        if self.nonlinear:
            y = self.sigma(y)

        if self.avg is not None:
            y = self.avg(y)

        return y
