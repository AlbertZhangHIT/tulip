import random
import time, datetime
import os, shutil
from math import sqrt
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
import torchnet as tnt
from torch.nn.modules.utils import  _pair
import ast
from tqdm import tqdm

import dataloader
import adversarial_training
import models
from models.utils import num_parameters
from penalties import LipschitzPenalty, TikhonovPenalty
import _helpers as helpers
from _parsers import parser
from utils import progress_bar

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Parse command line and miscellaneous set up
args = parser.parse_args()

args.has_cuda = torch.cuda.is_available()
args.data_workers=4

if args.log_dir is not None:
    os.makedirs(args.log_dir, exist_ok=True)

if args.seed is None:
    args.seed = int(time.time())
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = True

if args.has_cuda:
    args.num_gpus = torch.cuda.device_count()
else:
    args.num_gpus = 0

if args.log_dir is None:
    args.log_dir = os.path.join('./logs/',args.dataset,args.model,
            '{0:%Y-%m-%dT%H%M%S}'.format(datetime.datetime.now()))


print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')


# Data loaders
train_loader = getattr(dataloader, args.dataset)(args, train=True) # training data
test_loader = getattr(dataloader, args.dataset)(args, train=False) # test data

training_logger, testing_logger = helpers.loggers(args)

# Initialize model
Ntrain = train_loader.Ntrain
classes = train_loader.classes
in_channels = train_loader.in_channels
in_shape = train_loader.in_shape
model_args = ast.literal_eval(args.model_args)
model_args.update(bn=args.bn, dropout=args.dropout,
                  classes=classes, in_channels=in_channels,
                  kernel_size=args.kernel_size, sigmoid=args.tanh)
model = getattr(models, args.model)(**model_args)


# Loss function and regularizers
criterion = nn.CrossEntropyLoss()
tikhonov_penalty = TikhonovPenalty(model, bias=False)
lipschitz_penalty = LipschitzPenalty('2', create_graph=True)


# Move to GPU if available
if args.has_cuda:
    criterion = criterion.cuda(0)
    model = model.cuda(0)
    tikhonov_penalty = tikhonov_penalty.cuda(0)
    lipschitz_penalty = lipschitz_penalty.cuda(0)


# Optimizer and learning rate schedule
optimizer = optim.SGD(model.parameters(),
                      lr = args.lr,
                      momentum = args.momentum,
                      nesterov = args.nesterov)
scheduler = helpers.scheduler(optimizer,args)




# --------
# Training
# --------
decay = args.decay
lip = args.lip
def add_penalties(loss, data):

    if decay > 0:
        tik = tikhonov_penalty()
        loss = loss + decay * tik

    if lip >0:
        loss = loss + lip * lipschitz_penalty(loss, data)

    return loss

a_ = [args.J1>0, args.J2>0, args.Jinf>0]
assert sum(a_)<2, 'Only one adversarial training method may be active' 
if args.J1>0:
    perturb = adversarial_training.L1Perturbation(model, args.J1, criterion)
elif args.J2>0:
    perturb = adversarial_training.L2Perturbation(model, args.J2, criterion)
elif args.Jinf>0:
    perturb = adversarial_training.LInfPerturbation(model, args.Jinf, criterion)
elif args.PGDinf is not None :
    config = ast.literal_eval(args.PGDinf)
    perturb = adversarial_training.LinfPgdPerturbation(model, config, criterion)
else:
    perturb = lambda x, y: x
# config = {
#     'epsilon': 8.0 / 255, # Test 1.0-8.0
#     'num_steps': 10,
#     'step_size': 1.0 / 255, # 5.0
#     'random_start': True,
#     'loss_func': 'xent',
# }
# perturb = adversarial_training.LinfPgdPerturbation(model, config, criterion)

class TrainError(Exception):
    """Exception raised for error during training."""
    pass

def train(epoch, ttot):
    # Put the model in train mode (turn on dropout, unfreeze
    # batch norm parameters, make parameters differentiable)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)


    # Run through the training data
    if args.has_cuda:
        torch.cuda.synchronize()
    tepoch = time.perf_counter()
    el_loss = tnt.meter.AverageValueMeter()
    correct = 0; total = 0; train_loss = 0
    for batch_ix, (data, target) in enumerate(train_loader):

        if args.has_cuda:
            data, target = data.cuda(0), target.cuda(0)

        data = perturb(data, target)

        if args.lip>0:
            data.requires_grad_()
        else:
            data.requires_grad_(False)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss = add_penalties(loss, data)

        if np.isnan(loss.data.item()):
            raise TrainError('model returned nan during training')

        if args.lip>0:
            data.requires_grad_(False)

        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        progress_bar(batch_ix, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_ix+1), 100.*correct/total, correct, total))

        # Report training loss
        if (not (batch_ix % args.log_interval == 0 and batch_ix > 0)
            and (batch_ix +1) != Ntrain // args.batch_size ):
            loss = None
        if args.log_dir is not None:
            training_logger(loss,optimizer,tepoch,ttot)

        if (batch_ix % args.log_interval == 0 and batch_ix > 0):
            print('[Epoch %2d, batch %3d] penalized training loss: %.3g' %
                (epoch, batch_ix, loss.data.item()))
    if args.has_cuda:
        torch.cuda.synchronize()
    return ttot + time.perf_counter() - tepoch





def test(epoch, ttot):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():

        # Get the true training loss and error
        top1_train = tnt.meter.ClassErrorMeter()
        train_loss = tnt.meter.AverageValueMeter()
        for data, target in train_loader:
            if args.has_cuda:
                target = target.cuda(0,async=True)
                data = data.cuda(0,async=True)
            data.requires_grad_(True)
            data = perturb(data, target)
            output = model(data)


            top1_train.add(output.data, target.data)
            loss = criterion(output, target)
            train_loss.add(loss.data.item())

        t1t = top1_train.value()[0]
        lt = train_loss.value()[0]


        # Functions for tracking test loss, error
        test_loss = tnt.meter.AverageValueMeter()
        top1 = tnt.meter.ClassErrorMeter()

            
        # Evaluate test data
        for data, target in test_loader:
            if args.has_cuda:
                target = target.cuda(0,async=True)
                data = data.cuda(0,async=True)
            data = perturb(data, target)
            output = model(data)

            loss = criterion(output, target)
            

            top1.add(output, target)
            test_loss.add(loss.item())

        t1 = top1.value()[0]
        l = test_loss.value()[0]

        tik = tikhonov_penalty() if args.decay > 0  else 0.

    # Report results
    if args.log_dir is not None:
        testing_logger(epoch, l, t1, lt, t1t, ttot, tik, optimizer)

    print('[Epoch %2d] Average test loss: %.3f, error: %.2f%%'
            %(epoch, l, t1))
    print('%28s: %.3f, error: %.2f%%\n'
            %('training loss',lt,t1t))

    return test_loss.value()[0], top1.value()[0]





def main():

    save_path = args.log_dir if args.log_dir is not None else '.'

    # Save argument values to yaml file
    args_file_path = os.path.join(save_path, 'args.yaml')
    with open(args_file_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    save_model_path = os.path.join(save_path, 'checkpoint.pth.tar')
    best_model_path = os.path.join(save_path, 'best.pth.tar')

    pct_max = 100.*(1 - 1.0/classes)
    fail_count = 5
    time = 0.
    pct0 = 100.
    for e in range(args.epochs):

        # Update the learning rate
        scheduler.step()

        try:
            time = train(e, time)
        except TrainError:
            fail_count = 0

        loss, pct_err= test(e,time)
        if pct_err >= pct_max:
            fail_count -= 1

        torch.save({'epoch': e + 1,
                    'model': args.model,
                    'state_dict':model.state_dict(),
                    'pct_err': pct_err,
                    'loss': loss,
                    'optimizer': optimizer.state_dict()}, save_model_path)
        if pct_err < pct0:
            shutil.copyfile(save_model_path, best_model_path)
            pct0 = pct_err

        if fail_count < 1:
            break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupt; exiting')
