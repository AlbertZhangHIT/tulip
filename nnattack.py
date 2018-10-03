import os
import argparse
import numpy as np
import sys
from numpy import inf
import torch as th

import dataloader

parser = argparse.ArgumentParser('Nearest Distinct Neighbour attack and Lipschitz constant of data')

parser.add_argument('--batch-size', type=int, default=500,metavar='N',
        help='number of train images to check at a time')
parser.add_argument('--test-batch-size', type=int, default=500,metavar='N',
        help='number of test images to check at a time')
parser.add_argument('--dataset', type=str, default='cifar10', metavar='DS',
        choices=['cifar10','cifar100',  'TinyImageNet','Fashion'])
parser.add_argument('--data', type=str, required=True, metavar='DIR', 
        help='data storage directory')
parser.add_argument('--train-only',action='store_true',default=False,
        help='Compute distances between train images. Default is to '
        'instead compute distance of each test image to nearest train image')
args = parser.parse_args()

args.data_workers=4
args.has_cuda = th.cuda.is_available()
args.cutout=0




if args.train_only:
    bs = args.batch_size
    args.batch_size = args.test_batch_size
bs0 = args.test_batch_size
test_loader = getattr(dataloader, args.dataset)(args, train=args.train_only, transform=False)
if args.train_only:
    args.batch_size=bs
bs1 = args.batch_size
train_loader = getattr(dataloader, args.dataset)(args, train=True, transform=False)
nd = test_loader.in_shape**2 * test_loader.in_channels

if args.train_only:
    Nt = test_loader.Ntrain
else:
    Nt = 10000
dist = th.zeros(Nt)
if args.has_cuda:
    dist = dist.cuda()

for a, (xt,yt) in enumerate(test_loader):
    sys.stdout.write('Completed %6.2f%%\r'%(100*a/len(test_loader)))
    sys.stdout.flush()

    xt = xt.view(-1,nd)
    m = th.full(yt.size(),inf)
    if args.has_cuda:
        xt = xt.cuda()
        yt = yt.cuda()
        m = m.cuda()
    yts = yt.view(-1,1).expand(-1,bs1)

    for b, (x, y) in enumerate(train_loader):
        if args.has_cuda:
            x = x.cuda()
            y = y.cuda()
        ys = y.view(1,-1).expand(bs0,-1)
        bo = ys==yts

        x = x.view(-1,nd)
        se = xt[:,None,:] - x[None,:,:]
        d = se.norm(dim=2)
        d[bo]=inf
        d = d.min(dim=1)[0]
        m = th.min(d,m)
    i0 = a*bs0
    iN = i0 + len(yt)
    ix = th.arange(i0,iN)
    if args.has_cuda:
        ix = ix.cuda()
    dist[ix]=m
sys.stdout.write('Completed %6.2f%%\r'%(100.))
sys.stdout.flush()

s='tr-tr' if args.train_only else 'te-tr'
os.makedirs('nnattack/',exist_ok=True)
np.save('nnattack/'+args.dataset+'-'+s+'-dist.npy',dist.cpu().numpy())
print('\n\n%s Nearest Distinct Neighbour attack statistics:'%args.dataset)
print('mean l2: %.4g'%dist.mean())
print('median l2: %.4g'%dist.median())

# sum datasets have duplicate images, discard those with dist=0
print('Lipschitz 2,inf: %.4g'%(1/(dist[dist>0].min())))
