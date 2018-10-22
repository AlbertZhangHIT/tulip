"""Functions to adversarially perturb an input image during model training"""
from torch.autograd import grad
from math import sqrt, inf
import torch

def _prod(tup):
    p = 1
    for x in tup:
        p *= x
    return p

class Perturb(object):
    """Base class for all adversarial training perturbations"""

    def __init__(self, model, epsilon, criterion):
        """Initialize with a model, a maximum Euclidean perturbation 
        distance epsilon, and a criterion (eg the loss function)"""
        self.model = model
        self.eps = epsilon
        self.criterion = criterion

    def _prep_model(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _release_model(self):
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad_(True)

    def __call__(self, *args):
        raise NotImplementedError

class L1Perturbation(Perturb):
    """
    Penalize a loss function by the L1 norm of the loss's gradient.

    Equivalent to the Fast Gradient Signed Method.
    """

    def __init__(self, model, epsilon, criterion):
        super().__init__(model, epsilon, criterion)

    def __call__(self, x, y):
        self._prep_model()
        x.requires_grad_(True)
        Nd = _prod(x.shape[1:])

        yhat = self.model(x)
        loss = self.criterion(yhat, y)
        dx = grad(loss, x)[0]

        p = dx.sign() / sqrt(Nd)

        self._release_model()

        return x.detach() + self.eps*p

FGSM = L1Perturbation

class L2Perturbation(Perturb):
    """
    Penalize a loss function by the L2 norm of the loss's gradient.
    """

    def __init__(self, model, epsilon, criterion):
        super().__init__(model, epsilon, criterion)

    def __call__(self, x, y):
        self._prep_model()
        x.requires_grad_(True)

        yhat = self.model(x)
        loss = self.criterion(yhat, y)
        dx = grad(loss, x)[0]

        dxshape = dx.shape
        dx = dx.view(dxshape[0],-1)
        dxn = dx.norm(dim=1, keepdim=True)
        b = (dxn>0).squeeze()
        dx[b] = dx[b]/dxn[b]
        p = dx.view(*dxshape)

        self._release_model()

        return x.detach() + self.eps*p

class LInfPerturbation(Perturb):
    """
    Penalize a loss function by the L-infinity norm of the loss's gradient.
    """

    def __init__(self, model, epsilon, criterion):
        super().__init__(model, epsilon, criterion)

    def __call__(self, x, y):
        self._prep_model()
        x.requires_grad_(True)

        yhat = self.model(x)
        loss = self.criterion(yhat, y)
        dx = grad(loss, x)[0]

        dxshape = dx.shape
        bsz = dxshape[0]
        dx = dx.view(bsz,-1)
        ix = dx.abs().argmax(dim=-1)
        dx_ = torch.zeros_like(dx.view(bsz,-1))
        jx = torch.arange(bsz, device=dx.device)
        dx_[jx,ix] = dx.sign()[jx,ix]
        p = dx_.view(*dxshape)

        self._release_model()

        return x.detach() + self.eps*p
