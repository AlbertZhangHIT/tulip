import torch as th
from torch import nn
from torch.nn.modules.utils import _pair
from . import blocks

from .utils import View, num_parameters, Avg2d, Activation

class WResNet(nn.Module):

    def __init__(self, layers, factor, block='BasicBlock', in_channels=3,
                 classes=10, kernel_size=3, 
                 bn=False, dropout=0., sigmoid=False,
                 **kwargs):
        """A Wide ResNet for CIFAR10"""

        super().__init__()
        kernel_size = _pair(kernel_size)

        assert len(layers) == 3


        nl = 4
        for l in layers:
            nl += 2*l
        self.nl = nl
        self.factor = factor

        block = getattr(blocks, block)

        layer0 = [nn.Conv2d(in_channels, 16, kernel_size, padding=1, bias = True)]
        if bn:
            layer0.append(nn.BatchNorm2d(16, affine=False))
                
        layer0.append(Activation())
        self.layer0 = nn.Sequential(*layer0)

        def make_layer(count, in_channels, out_channels, stride):
            return nn.Sequential(
                      block(in_channels, out_channels, stride=stride,
                            kernel_size=kernel_size, bn=bn, dropout=dropout,
                            **kwargs),
                      *[block(out_channels, out_channels, stride=1,
                            kernel_size=kernel_size, bn=bn, dropout=dropout,
                            **kwargs) 
                        for _ in range(count-1)] )

        self.layer1 = make_layer(layers[0], 16, 16*factor, 1)
        self.layer2 = make_layer(layers[1], 16*factor, 32*factor, 2)
        self.layer3 = make_layer(layers[2], 32*factor, 64*factor, 2)

        self.pool = Avg2d()
        self.view = View(64*factor)

        self.fc = nn.Linear(64*factor, classes, bias=True)

        self.sigmoid = sigmoid
        if sigmoid:
            self.bn = nn.BatchNorm1d(classes, affine=False)
            self.t = nn.Parameter(th.tensor(1.))

        self.num_parameters = num_parameters(self)
        self.classes=classes
        self.nonlinear=True

        print('[WResNet%d-%d] Num parameters: %.2e' %(self.nl, factor, num_parameters(self)))

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.view(x)
        x = self.fc(x)
        if self.sigmoid:
            x = self.bn(x)
            if self.nonlinear:
                x = th.tanh(x/self.t.abs())
                x = self.t.abs() * x

        return  x


    def extra_repr(self):
        s = ('layers={nl}, widening factor={factor}, # params={num_parameters:.3e}')
        return s.format(**self.__dict__)

def WResNet28_10(**kwargs):
    return WResNet([4,4,4],10,**kwargs)

def WResNet16_8(**kwargs):
    return WResNet([2,2,2],8,**kwargs)

def WResNet16_4(**kwargs):
    return WResNet([2,2,2],4,**kwargs)

def WResNet22_4(**kwargs):
    return WResNet([3,3,3],4,**kwargs)

def WResNet40_4(**kwargs):
    return WResNet([6,6,6],4,**kwargs)
