import os
import numpy as np
import torch as th
import torchvision 
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST 


def cutout(mask_size,channels=3):
    if channels>1:
        mask_color=tuple([0]*channels)
    else:
        mask_color=0

    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    if mask_size >0:
        def _cutout(image):
            image = np.asarray(image).copy()

            if channels >1:
                h, w = image.shape[:2]
            else:
                h, w = image.shape

            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

            cx = np.random.randint(cxmin, cxmax)
            cy = np.random.randint(cymin, cymax)
            xmin = cx - mask_size_half
            ymin = cy - mask_size_half
            xmax = xmin + mask_size
            ymax = ymin + mask_size
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            image[ymin:ymax, xmin:xmax] = mask_color
            if channels==1:
                image = image[:,:,None]
            return image
    else:
        def _cutout(image):
            return image

    return _cutout



def TinyImageNet(args, train=True, transform=True):

    dataset_dir = os.path.join(args.data,'TinyImageNet')

    kwargs = {'num_workers': args.data_workers, 'pin_memory': args.has_cuda}


    if train and transform:
        d = os.path.join(dataset_dir,'train')
        train_dataset = torchvision.datasets.ImageFolder(d, transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        cutout(args.cutout),
                        transforms.ToTensor()]))

        ds = th.utils.data.DataLoader(train_dataset, batch_size =
                args.batch_size, shuffle = True, drop_last=True,
                **kwargs)


    else:
        if not train:
            d = os.path.join(dataset_dir,'val')
        else:
            d = os.path.join(dataset_dir,'train')
        test_dataset = torchvision.datasets.ImageFolder(d,
                transforms.Compose([transforms.ToTensor()]))
        bs = args.batch_size if train else args.test_batch_size
        ds  = th.utils.data.DataLoader(test_dataset, batch_size = bs,
                shuffle = False,
                drop_last = False, **kwargs  )


    ds.Ntrain = 100000
    ds.classes = 200
    ds.in_channels = 3
    ds.in_shape = 64

    return ds

def Fashion(args,train=True, prob = 0.1, transform=True):

    if train and transform:
        tlist = [transforms.RandomCrop(28,padding=4),
                 transforms.RandomHorizontalFlip(),
                 cutout(args.cutout, channels=1),
                 transforms.ToTensor()]
    else:
        tlist = [transforms.ToTensor()]


    transform = transforms.Compose(tlist)
    root = os.path.join(args.data,'FashionMNIST')

    ds = FashionMNIST(root, download=True, train=train, transform=transform)


    bs = args.batch_size if train else args.test_batch_size

    dataloader = th.utils.data.DataLoader(ds,
                      batch_size = bs,
                      drop_last = True if (train and transform) else False,
                      shuffle = True if (train and transform) else False,
                      num_workers = args.data_workers,
                      pin_memory = args.has_cuda)

    dataloader.Ntrain = 60000
    dataloader.classes = 10
    dataloader.in_channels = 1
    dataloader.in_shape = 28

    return dataloader


def cifar10(args,train=True, transform=True):
    if train and transform:
        tlist = [transforms.RandomCrop(32,padding=4),
                 transforms.RandomHorizontalFlip(),
                 cutout(args.cutout),
                 transforms.ToTensor()]
    else:
        tlist = [transforms.ToTensor()]




    transform = transforms.Compose(tlist)
    root = os.path.join(args.data,'cifar10')


    ds = CIFAR10(root, download=True, train=train, transform=transform)


    bs = args.batch_size if train else args.test_batch_size

    dataloader = th.utils.data.DataLoader(ds,
                      batch_size = bs,
                      drop_last = True if (train and transform) else False,
                      shuffle = True if (train and transform) else False,
                      num_workers = args.data_workers,
                      pin_memory = args.has_cuda)

    dataloader.Ntrain = 50000
    dataloader.classes = 10
    dataloader.in_channels =  3
    dataloader.in_shape = 32


    return dataloader


def cifar100(args,train=True, transform=True):

    if train and transform:
        tlist = [transforms.RandomCrop(32,padding=4),
                 transforms.RandomHorizontalFlip(),
                 cutout(args.cutout),
                 transforms.ToTensor()]
    else:
        tlist = [transforms.ToTensor()]
    transform = transforms.Compose(tlist)

    root = os.path.join(args.data,'cifar100')

    ds = CIFAR100(root, download=True, train=train,transform=transform)

    bs = args.batch_size if train else args.test_batch_size

    dataloader = th.utils.data.DataLoader(ds,
                      batch_size = bs,
                      drop_last = True if (train and transform) else False,
                      shuffle = True if (train and transform) else False,
                      num_workers = args.data_workers,
                      pin_memory = args.has_cuda)
    dataloader.Ntrain = 50000
    dataloader.classes = 100
    dataloader.in_channels = 3
    dataloader.in_shape = 32

    return dataloader
