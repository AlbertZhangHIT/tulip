import os
import time
import ast
import bisect
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import  SGD
import torch.nn as nn

def scheduler(optimizer,args):
    """Return a hyperparmeter scheduler for the optimizer"""
    lS = np.array(ast.literal_eval(args.lr_schedule))
    llam = lambda e: float(lS[max(bisect.bisect_right(lS[:,0], e)-1,0),1])
    lscheduler = LambdaLR(optimizer, llam)

    return lscheduler

def loggers(args):
    """Training and test loggers from input arguments"""
    os.makedirs(args.log_dir, exist_ok=True)

    training_log = open(os.path.join(args.log_dir, 'training.csv'), 'w')
    testing_log = open(os.path.join(args.log_dir, 'test.csv'), 'w')

    print('%s,%s,%s,%s'% ('index','time', 'fval', 'learning rate'),
            file=training_log)
    print('%s,%s,%s,%s,%s,%s,%s,%s'%('epoch','time','fval','pct_err',
          'train_fval','train_pct_err',
          'tikhonov', 
          'learning rate'),
            file=testing_log)



    class training_logger(object):
        def __init__(self):
            self.index = 0
        def __call__(self, loss, optimizer, tepoch,ttot):
            lr_current = optimizer.param_groups[0]["lr"]
            if loss is not None:
                print('%d,%.7g,%.3e,%.3e'% (self.index, time.time() - tepoch + ttot, loss.data.item(),lr_current),
                    file=training_log, flush=True)
            self.index+=1


    class testing_logger(object):
        def __init__(self):
            super(testing_logger,self).__init__()

        def __call__(self, epoch,loss,top1,trainl,top1l, time,tik, 
                optimizer ):
            lr_current = optimizer.param_groups[0]["lr"]
            print('%d,%.7g,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e'%
                    (epoch,time,loss,top1,trainl,top1l,tik,
                         lr_current),
                    file=testing_log, flush=True)

    return training_logger(), testing_logger()
