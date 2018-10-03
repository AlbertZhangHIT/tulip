import warnings
import argparse
import os
import yaml
import ast
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchnet as tnt

import models
from penalties import JacobianPenalty, LipschitzPenalty
from penalties import TikhonovPenalty, MatrixProductPenalty
import dataloader




parser = argparse.ArgumentParser('Report summary statistics on a trained model')

parser.add_argument('--model-dir', type=str, default=None, required=True, metavar='DIR',
        help='Directory where model is saved')
parser.add_argument('--batch-size', type=int, default=100,metavar='N',
        help='number of images to attack at a time')


args = parser.parse_args()

state_dict = yaml.load(open(os.path.join(args.model_dir, 'args.yaml'), 'r'))
state = argparse.Namespace()
state.__dict__ = state_dict

sigmoid = state.tanh


args.has_cuda = torch.cuda.is_available()
batch_size = args.batch_size
print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

state.__dict__['test_batch_size'] = batch_size

# get the data
test_loader = getattr(dataloader, state.dataset)(state, train=False)

# load the model
model_args = ast.literal_eval(state.model_args)

model_args.update(bn=state.bn,dropout=state.dropout,
        classes=test_loader.classes, in_channels=test_loader.in_channels,
        kernel_size=state.kernel_size,
        sigmoid=sigmoid)

m = getattr(models,state.model)(**model_args)

print('\n')
m_pth = os.path.join(args.model_dir, 'best.pth.tar')

pth = torch.load(m_pth, map_location='cpu')
m.load_state_dict(pth['state_dict'], strict=True)
try:
    t = m.t
except:
    t = np.nan
m.eval()
if args.has_cuda:
    m = m.cuda()

jl = JacobianPenalty('2,inf', retain_graph=True)
gl = LipschitzPenalty('2', create_graph=True)
l2p = TikhonovPenalty(m, bias=False)
shape = (test_loader.in_channels, test_loader.in_shape, test_loader.in_shape)
mpp = MatrixProductPenalty(shape, m, '2,inf', bsz = args.batch_size)


CE = nn.CrossEntropyLoss()

if args.has_cuda:
    CE = CE.cuda()
    gl = gl.cuda()
    jl = jl.cuda()
    l2p = l2p.cuda()
    mpp = mpp.cuda()


jacobianloss  = 0.
cegradloss = 0.
for p in m.parameters():
    p.requires_grad_(False)

CEloss = tnt.meter.AverageValueMeter()
top1 = tnt.meter.ClassErrorMeter()


minconfidence = np.inf
for i, (data, target) in enumerate(test_loader):
    if args.has_cuda:
        data = data.cuda().requires_grad_()
        target = target.cuda()
    y = m(data)

    
    xsort = y.softmax(dim=1).sort(dim=-1)
    xam = xsort[1][:,-1]
    ixcorrect = xam==target
    confidence = xsort[0][ixcorrect,-1] - xsort[0][ixcorrect,-2]
    minconfidence = min(confidence.min(), minconfidence)


    celoss = CE(y, target)

    CEloss.add(celoss.item())
    n = y.shape[0]
    top1.add(y.data, target)

    gl_ = gl(celoss, data)
    cenormgrad = gl_

    jl0 = jl(data, y)
    jacobianloss  = max(jacobianloss, jl0)
    cegradloss = max(cenormgrad, cegradloss)



wpn = mpp().item()
conf = minconfidence.item()
print('Performance statistics on test data')
print('-----------------------------------')
stats = {'pct_err': top1.value()[0],
         'epoch': pth['epoch'],
         'Cross-entropy loss ': CEloss.value()[0],
         'Jacobian 2,inf norm': jacobianloss.item() , 
         'Weight product 2,inf norm': wpn,
         'Min correct test confidence': conf,
         'adversarial distance lower bound': conf/(2*wpn),
         'max norm CE loss grad': cegradloss.item() ,
         'l2 weights': l2p().sqrt().item(),
         'tanh scale factor': t}

for k, v in stats.items():
    print('%25s: %.3g'%(k,v))

outfile = 'summary.out'
pickle.dump(stats,open(os.path.join(args.model_dir,'summary.p'),'wb'))

with open(os.path.join(args.model_dir,outfile),'w') as f:
    for i, (k, v) in enumerate(stats.items()):
        f.write('%25s: %.3e\n'%(k,v))
