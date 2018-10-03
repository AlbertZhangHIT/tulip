import torch as th
from torch import nn
from torch.nn.modules.utils import _pair
from . import blocks

from .utils import View, num_parameters, Avg2d, Activation



class ResNet(nn.Module):

    def __init__(self, layers, block='BasicBlock', in_channels=3,
                 classes=10, kernel_size=3,
                 bn=False, dropout=0., sigmoid = False,
                 **kwargs):
        """A ResNet for CIFAR10"""

        super().__init__()
        kernel_size = _pair(kernel_size)

        assert len(layers) == 3

        nl = 4 if not bn else 7
        for l in layers:
            if bn:
                nl += 2*2*l
            else:
                nl += 2*l
        self.nl = nl

        try:
            base_channels=kwargs['base_channels']
        except Exception as e:
            base_channels=16


        self.blocktype=block
        block = getattr(blocks, block)

        layer0 = [nn.Conv2d(in_channels, base_channels, kernel_size,
                    padding=1, bias = True)]
        if bn:
            layer0.append(nn.BatchNorm2d(base_channels, affine=False))
        layer0.append(Activation())
        self.layer0 = nn.Sequential(*layer0)

        def make_layer(count, in_channels, out_channels, stride):
            return nn.Sequential(
                      block(in_channels, out_channels, stride=stride,
                            kernel_size=kernel_size, bn=bn, dropout=dropout,
                            **kwargs),
                      *[block(out_channels, out_channels, stride=1,
                            kernel_size=kernel_size, bn=bn, dropout=dropout,
                            **kwargs) for _ in range(count-1)] )

        self.layer1 = make_layer(layers[0], base_channels, base_channels, 1)
        self.layer2 = make_layer(layers[1], base_channels, base_channels*2, 2)
        self.layer3 = make_layer(layers[2], base_channels*2, base_channels*4, 2)

        self.pool = Avg2d()
        self.view = View(4*base_channels)

        self.fc = nn.Linear(4*base_channels, classes, bias=True)

        self.sigmoid = sigmoid
        if sigmoid:
            self.bn = nn.BatchNorm1d(classes, affine=False)
            self.t = nn.Parameter(th.tensor(1.))
        self.classes=classes
        self.nonlinear=True

        self.num_parameters=num_parameters(self)

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

        return x 


    def extra_repr(self):
        s = ('layers={nl}, # params={num_parameters:.3e}, block={blocktype}')
        return s.format(**self.__dict__)

def ResNet22(**kwargs):
    m = ResNet([3,3,3],**kwargs)
    print('[ResNet22] Num parameters: %.2e' %(m.num_parameters))
    return m

def ResNet34(**kwargs):
    m = ResNet([5,5,5],**kwargs)
    print('[ResNet34] Num parameters: %.2e' %(m.num_parameters))
    return m

def ShakeShake34(**kwargs):
    m = ResNet([5,5,5],block='BranchBlock',method='shake',**kwargs)
    print('[ShakeShake%d] Num parameters: %.2e' %(m.num_parameters))
    return m

def ResNeXt34(**kwargs):
    m = ResNet([5,5,5],block='BranchBlock',
            base_channels=32,**kwargs)
    print('[ResNeXt34] Num parameters: %.2e' %(m.num_parameters))
    return m
