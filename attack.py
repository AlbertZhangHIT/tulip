import argparse
import yaml
import os, sys
import ast
from warnings import warn
import contextlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import foolbox

import models
import dataloader


parser = argparse.ArgumentParser('Test the stability of a model using FoolBox')

parser.add_argument('--model-dir', type=str, required=True,metavar='DIR',
        help='Directory where model is saved')
parser.add_argument('--num-images', type=int, default=10000,metavar='N',
        help='total number of images to attack')
parser.add_argument('--batch-size', type=int, default=100,metavar='N',
        help='number of images to attack at a time')
parser.add_argument('--norm-type', type=str, default='L2',metavar='NORM', 
        choices=['L2','Linf'])
parser.add_argument('--epsilon',nargs='+',type=float,metavar='EPS',
        default = [0,1e-4,1e-3,1e-2,1e-1,1,10,100],
        help='Report a summary of attacks with these magnitudes')
parser.add_argument('--attack-type', type=str, default='gradient',
        choices=['deepfool','iterative-gradient','gradient','boundary','fgsm','ifgsm'],
        help='Attack type (default: gradient)')


args = parser.parse_args()

state_dict = yaml.load(open(os.path.join(args.model_dir, 'args.yaml'), 'r'))
state = argparse.Namespace()
state.__dict__ = state_dict


args.has_cuda = torch.cuda.is_available()
bsz = args.batch_size




# Get data
state.test_batch_size = bsz
state.batch_size = bsz
state.has_cuda = args.has_cuda

loader = getattr(dataloader, state.dataset)(state, train=False, transform=False)
ds = loader.dataset
c, h, w = ds[0][0].size()
num_pixels = c*h*w

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

# Load model
classes = loader.classes
in_channels = loader.in_channels

model_args = ast.literal_eval(state.model_args)
sigmoid=state.tanh
model_args.update(bn=state.bn,dropout=state.dropout,
        classes=classes, in_channels=in_channels,
        kernel_size=state.kernel_size,
        sigmoid=sigmoid)

m = getattr(models,state.model)(**model_args)

m_pth = os.path.join(args.model_dir, 'best.pth.tar')

m.load_state_dict(torch.load(m_pth,
                             map_location='cpu')['state_dict'],
                  strict=True)
m.eval()

if torch.cuda.is_available():
    m = m.cuda()

for p in m.parameters():
    p.requires_grad_(False)

pmodel = m

#Fool box interface
fmodel = foolbox.models.PyTorchModel(pmodel, bounds=(0,1), num_classes=classes)
attack_criteria = foolbox.criteria.Misclassification()
if args.norm_type=='L2' and args.attack_type=='deepfool':
    attack = foolbox.attacks.DeepFoolAttack(model=fmodel, criterion=attack_criteria)
elif args.norm_type=='Linf' and args.attack_type=='deepfool':
    attack = foolbox.attacks.DeepFoolLinfinityAttack(model=fmodel, criterion=attack_criteria)
elif args.attack_type=='gradient':
    attack = foolbox.attacks.GradientAttack(model=fmodel, criterion=attack_criteria)
elif args.attack_type=='iterative-gradient' and args.norm_type=='L2':
    attack = foolbox.attacks.L2BasicIterativeAttack(model=fmodel, criterion=attack_criteria)
elif args.attack_type=='iterative-gradient' and args.norm_type=='Linf':
    attack = foolbox.attacks.LinfinityBasicIterativeAttack(model=fmodel, criterion=attack_criteria)
elif args.attack_type=='boundary':
    attack = foolbox.attacks.BoundaryAttack(model=fmodel, criterion=attack_criteria)
elif args.attack_type=='fgsm':
    attack = foolbox.attacks.GradientSignAttack(model=fmodel, criterion=attack_criteria)
elif args.attack_type=='ifgsm':
    attack = foolbox.attacks.LinfinityBasicIterativeAttack(model=fmodel, criterion=attack_criteria)

d = np.zeros(args.num_images)
for i, (x,y) in enumerate(loader):
    for j in range(len(y)):
        if i*args.batch_size + j >= args.num_images:
            break
        if j==0:
            sys.stdout.write('[%6.2f%%]\r'%(100*i*bsz/args.num_images))
            sys.stdout.flush()
        Im = x[j].numpy()
        La = y[j].item()

        adversarial = foolbox.adversarial.Adversarial(fmodel, attack_criteria, Im, La)
        if not args.attack_type=='boundary':
            attack(adversarial)
        else:
            with open(os.devnull,'w') as f:
                with contextlib.redirect_stdout(f):
                    attack(adversarial, iterations=100)
        l2 = np.sqrt(num_pixels*adversarial.distance.value)

        d[i*args.batch_size + j] = l2

d = np.sort(d)
st = args.attack_type
if args.attack_type in ['deepfool', 'iterative-gradient']:
    st = st + '-'+args.norm_type

np.save(os.path.join(args.model_dir,'foolbox-'+st+'-dist.npy'),d)

n = [(d<=e).sum()*100/args.num_images for e in args.epsilon]

df = pd.DataFrame({'epsilon':args.epsilon, 'pct err':n})
attack_log = open(args.model_dir + '/foolbox-'+st+'-dist.out', 'w')
print(df, file=attack_log)
print(df)
