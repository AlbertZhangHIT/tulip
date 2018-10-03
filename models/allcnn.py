import torch as th
import torch.nn as nn

from .utils import View, num_parameters, Avg2d, Id, Activation

class AllCNN(nn.Module):
    def __init__(self, bn=False, in_channels=3,
            classes=10, dropout=0., c1=96, c2=192,
            sigmoid=False, **kwargs):
        """Implementation of AllCNN-C [1].

           [1] Springenberg JT, Dosovitskiy A, Brox T, Riedmiller M. Striving
               for simplicity: The all convolutional net. arXiv preprint
               arXiv:1412.6806.  2014 Dec 21."""
        super().__init__()
        self.nl = 9 if not bn else 18

        def convbn(ci,co,ksz,s=1,pz=0,bn=False):
            conv = nn.Conv2d(ci,co,ksz, stride=s, padding=pz,
                              bias=True)

            if not bn:
                bn = Id()
            else:
                bn = nn.BatchNorm2d(co, affine=False)


            m = nn.Sequential(
                conv,
                bn,
                Activation())

            return m


        self.m = nn.Sequential(
            convbn(in_channels,c1,3,1,1,bn=bn),
            convbn(c1,c1,3,1,1,bn=bn),
            convbn(c1,c1,3,2,1,bn=bn),
            nn.Dropout(dropout),
            convbn(c1,c2,3,1,1,bn=bn),
            convbn(c2,c2,3,1,1,bn=bn),
            convbn(c2,c2,3,2,1,bn=bn),
            nn.Dropout(dropout),
            convbn(c2,c2,3,1,1,bn=bn),
            convbn(c2,c2,3,1,1,bn=bn),
            convbn(c2,classes,1,1,bn=bn),
            Avg2d(),
            View(classes))


        self.sigmoid = sigmoid
        if sigmoid:
            self.bn = nn.BatchNorm1d(classes, affine=False)
            self.t = nn.Parameter(th.tensor(1.))

        self.classes=classes
        self.nonlinear=True

        s = '[%s] Num parameters: %.2e'%(self.name, num_parameters(self.m))
        print(s)

    @property
    def name(self):
        return self.__class__.__name__

    def forward(self, x):
        x = self.m(x)
        if self.sigmoid:
            x = self.bn(x)
            if self.nonlinear:
                x = th.tanh(x/self.t.abs())
                x = self.t.abs() * x

        return x
