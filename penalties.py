import torch.nn as nn
import torch as th
from torch.autograd import Function, grad
from math import sqrt

def prod(l):
    p = 1
    for x in l:
        p*=x
    return p

class LipschitzPenalty(nn.Module):
    def __init__(self, norm, summary=th.max, create_graph=False):
        super().__init__()
        self.norm = norm
        self.create_graph=create_graph
        self.summary = summary

    def forward(self, l, x):
        sh = x.shape
        bsz = sh[0]

        if not x.requires_grad:
            x.requires_grad_()

        dx, = grad(l, x, create_graph=self.create_graph)
        dx = dx.view(bsz, -1)
        
        n2 = dx.norm(p=2,dim=-1)
        n1 = dx.norm(p=1,dim=-1)
        N = dx.shape[1]

        if self.norm in [2,'2']:
            n = n2
        elif self.norm in [1,'1']:
            n = n1
        elif self.norm=='inf':
            n = dx.abs().max(dim=-1)[0]

        return self.summary(n)


class JacobianPenalty(nn.Module):
    def __init__(self, norm, create_graph=False, retain_graph=False):
        super().__init__()
        self.norm = norm
        self.summary = th.max
        self.create_graph=create_graph
        self.retain_graph=retain_graph

    def forward(self, x, y, labels=None):
        xsh = x.shape
        ysh = y.shape
        bsz = xsh[0]
        assert bsz == ysh[0], 'Batch size of inputs and outputs do not agree'

        if labels is None:
            Jacobian = x.new_zeros(bsz, prod(ysh[1:]), prod(xsh[1:]))
            ys = y.sum(dim=0).view(-1)

            if not x.requires_grad:
                x.requires_grad_()


            for i, y in enumerate(ys): 
                if self.create_graph:
                    dX, = grad(y, x, create_graph=self.create_graph)
                else:
                    flag=False if i==(prod(ysh[1:])-1) else True
                    flag=flag or self.retain_graph
                    dX, = grad(y, x, retain_graph=flag)
                Jacobian[:, i, :] = dX.view(bsz, -1)

            if self.norm == '2,inf':
                norms = Jacobian.norm(p=2, dim=2).max(dim=1)[0]
            elif self.norm in ['inf,inf', 'inf']:
                norms = Jacobian.norm(p=1, dim=2).max(dim=1)[0]
            else:
                raise ValueError('%s is not an available norm'%self.norm)

            return self.summary(norms)
        else:
            i = th.arange(y.shape[0], dtype=th.long)
            ys = y[i,labels].sum()

            dX, = grad(ys, x, create_graph=self.create_graph,
                    retain_graph=self.retain_graph)
            dX = dX.view(bsz,-1)

            if self.norm in ['2,inf','2',2]:
                norms = dX.norm(p=2)
            else:
                raise ValueError('%s is not an available norm'%self.norm)

            return self.summary(norms)
            

class TikhonovPenalty(nn.Module):

    def __init__(self, model, bias=False, weight=True, others=True, cutoff=0.):
        super().__init__()
        self.model = model
        self.bias= bias
        self.others= others
        self.weight=weight
        self.cutoff = cutoff

    def forward(self):
        p = 0.
        n = 0
        for name, param in self.model.named_parameters():
            param = param.view(-1)

            if 'bias' in name and self.bias:
                p = p + param.norm()**2
            elif 'weight' in name and self.weight:
                p = p + param.norm()**2
            elif self.others:
                p = p + param.norm()**2

        return max(p,self.cutoff)


class MMt2InfNorm(Function):

    @staticmethod
    def forward(ctx, M):
        d = th.einsum('ij,ij->i',(M,M))
        i = d.argmax()
        n2 = d[i]
        n = n2.sqrt()

        ctx.save_for_backward(M,i,n)

        return n

    @staticmethod
    def backward(ctx, grad_output):
        M, i, n = ctx.saved_tensors
        dM = th.zeros_like(M)
        dM[i,:] = M[i,:].div(n)

        return dM * grad_output

mmt2infnorm = MMt2InfNorm.apply

class MatrixProductPenalty(nn.Module):
    def __init__(self, inshape, model, norm, bsz=100):
        super().__init__()
        self.model = model
        self.norm = norm
        self.inshape = inshape
        self._cuda=False
        self.bsz = bsz


    def cuda(self, device=None):
        self._cuda=True
        return self._apply(lambda t: t.cuda(device))

    def cpu(self):
        self._cuda=False
        return self._apply(lambda t: t.cpu())

    def forward(self):
        with th.no_grad():
            switch = False
            if self.model.training:
                switch = True
                self.model.eval()

            for layer in self.model.modules():
                try:
                    a = layer.nonlinear
                    layer.nonlinear = False
                except AttributeError as e:
                    pass

            din = prod(self.inshape)

            X = th.eye(din)
            Z = th.zeros(1,*self.inshape)
            if self._cuda:
                X = X.cuda()
                Z = Z.cuda()
            X = X.view(din, *self.inshape)

            
            xs = X.chunk(din//self.bsz)
            ys = [self.model(x) for x in xs]
            y = th.cat(ys)
            b = self.model(Z)
            #b.detach_()
            M = y - b

            M = M.transpose(0,1).view(-1, din)


            if self.norm == '2,inf':
                matrix_norm = mmt2infnorm(M)
            else:
                raise ValueError('%s is not an available norm'%self.norm)

            for layer in self.model.modules():
                try:
                    a = layer.nonlinear
                    layer.nonlinear = True
                except AttributeError as e:
                    pass

            if switch:
                self.model.train()

        return matrix_norm
