"""This module parses all command line arguments to main.py"""
import argparse
import numpy as np

parser = argparse.ArgumentParser('Testbed for DNN optimization in PyTorch')
parser.add_argument('--data', type=str, required=True, metavar='DIR',
        help='data storage directory')
parser.add_argument('--dataset', type=str, default='cifar10', metavar='DS',
        choices=['cifar10','cifar100', 'TinyImageNet','Fashion'])
parser.add_argument('--data-workers', type=int, default=4, metavar='N',
        help='number of data loading workers. (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
        help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--log-dir', type=str, default=None,metavar='DIR',
        help='directory for outputting log files. (default: ./logs/DATASET/MODEL/TIMESTAMP/')
parser.add_argument('--seed', type=int, default=None, metavar='S',
        help='random seed (default: int(time.time()) )')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
        help='number of epochs to train (default: 200)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)')

group1 = parser.add_argument_group('Model hyperparameters')
group1.add_argument('--model', type=str, default='ResNet34',
        help='Model architecture (default: ResNet34)')
group1.add_argument('--dropout',type=float, default=0, metavar='P',
        help = 'Dropout probability, if model supports dropout (default: 0)')
group1.add_argument('--cutout',type=int, default=0, metavar='N',
        help = 'Cutout size (default: 0)')
group1.add_argument('--bn',action='store_true', dest='bn',
        help = "Use batch norm (default)")
group1.add_argument('--no-bn',action='store_false', dest='bn',
       help = "Don't use batch norm")
group1.set_defaults(bn=True)
group1.add_argument('--kernel-size',type=int, default=3, metavar='K',
        help='convolution kernel size')
group1.add_argument('--model-args',type=str, default='{}',metavar='ARGS',
        help='A dictionary of extra arguments passed to the model. (default: "{}")')
group1.add_argument('--tanh',action='store_true', default=False,
       help = "Append tanh normalization layer to model output before softmax")


group0 = parser.add_argument_group('Optimizer hyperparameters')
group0.add_argument('--batch-size', type=int, default=128, metavar='N',
        help='Input batch size for training. (default: 128)')
group0.add_argument('--lr', type=float, default=0.1, metavar='LR',
        help='Initial step size. (default: 0.1)')
group0.add_argument('--lr-schedule', type=str, metavar='[[epoch,ratio]]',
        default='[[0,1],[60,0.2],[120,0.04],[160,0.08]]', help='List of epochs and multiplier '
        'for changing the learning rate (default: [[0,1],[60,0.2],[120,0.04],[160,0.08]]). ')
group0.add_argument('--momentum', type=float, default=0.9, metavar='M',
       help='SGD momentum parameter (default: 0.9)')
group0.add_argument('--nesterov', action='store_true', default=False,
        help='enable Nesterov momentum in SGD (default: False)')



group2 = parser.add_argument_group('Regularizers')
group2.add_argument('--lip', type=float, default=0., metavar = 'L',
        help='Lagrange multiplier for Lipschitz penalty (default: 0)')
group2.add_argument('--decay',type=float, default=0., metavar='L',
        help='Lagrange multiplier for weight decay (Tikhonov '
        'regularization) (default: 0)')
group2.add_argument('--J1', type=float, default=0.,
        metavar='L',
        help='Adversarial training step size, '
        'in signed gradient direction. Equivalent '
        'to penalizing the loss by the average L1 norm '
        'of the gradient wrt the images. (default: 0.)')
group2.add_argument('--J2', type=float, default=0.,
        metavar='L',
        help='Adversarial training step size, '
        'in gradient ascent direction. Equivalent '
        'to penalizing the loss by the average L2 norm '
        'of the gradient wrt the images. (default: 0.)')
group2.add_argument('--Jinf', type=float, default=0.,
        metavar='L',
        help='Penalize the loss by the average Linf norm '
        'of the gradient wrt the images. (default: 0.)')
