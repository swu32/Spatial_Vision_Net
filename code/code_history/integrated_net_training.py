#training function that merges previous repeated codes

import argparse
import os
import random
import shutil
import time
import warnings
import sys
from datetime import date
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
import model as models
from datetime import date

this_net = "SV_net_II"
# this_net “baseline_net”,"SV_net_I","SV_net_I_low_frequency","SV_net_II"
'''
all the net architecture to choose from: 

"baseline_net": a resnet18 implemented on ImageNet
correspond to.
ImageNet_baseline_model_best.pth &
ImageNet_baseline_checkpoint.pth 

"SV_net_I: first version of spatial vision net, with Spatial vision part as front end and resnet18 as backend,
correspond to 
ImageNet_normalization_model_best.pth.tar, & 
ImageNet_normalization_checkpoint.pth.tar"

"SV_net_I_low_frequency": same thing with SV_net_I, but employing only the lower half of the frequency filters,
correspond to
ImageNet_low_freq_model_best.pth & 
ImageNet_low_freq_checkpoint.pth & 

"SV_net_II": A simplified and a more updated version of SV_net_I. The spatial vision frontend has filter responses 
separated between positive and negative, and the spatial vision backend has similar structure instead of Resnet18, for 
the purpose of overcoming the overfitting behavior of the resnet18 backend. 
ImageNet_SV_net_II_model_best.pth.tar &
ImageNet_SV_net_II_model_checkpoint.pth.tar

"SV_net_II_low_frequency": 
ImageNet_SV_net_II_low_frequency_model_best.pth.tar &
ImageNet_SV_net_II_low_frequency_model_checkpoint.pth.tar


'''

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Set up training parameters')
parser.add_argument('data', metavar='DIR', default='/gpfs01/bethge/data/imagenet-raw/raw-data',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')


best_acc1 = 0  # this is a global variable here


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.world_size = 1

    ngpus_per_node = torch.cuda.device_count()

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    global best_acc1

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        # TODO: implement intergrated net across 12 different frequencies.
        print("=> creating model '{}'".format(args.arch))
        print('batch size is ', args.batch_size)
        if this_net == "SV_net_II":  # make sure the simplenet is working for pilot training.
            print('Employing simple net')
            model = models.simple_net(n_freq=12, n_orient=8, n_phase=2, imsize=224, num_classes=1000)
        elif this_net == "SV_net_II_low_frequency":
            print('Employing simple net with low frequency')
            model = models.low_freq_simple_net(n_freq=12, n_orient=8, n_phase=2, imsize=224, num_classes=1000)

    record_file_name = 'ImageNet_performance_record' + this_net + '0715.npy'

    # this_net “baseline_net”,"SV_net_I","SV_net_I_low_frequency","SV_net_II"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model = torch.nn.DataParallel(model).to(device)
    # define loss function (criterion) and optimizer
    # TODO: modify loss criterion to be compatable with independent filter responses
    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['accuracy']
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # best_acc1 may be from a checkpoint from a different GPU
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    normalize_baseline = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if this_net == "baseline_net":

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),  # Strange random RandomResizedCrop(224,scale = (0.08,0.50)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_baseline,
            ]))
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        """TODO: add gray values"""
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),  # Strange random RandomResizedCrop(224,scale = (0.08,0.50)),
                transforms.CenterCrop(224),
                transforms.Grayscale(),
                transforms.ToTensor(),
                normalize_baseline,
            ])), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)



    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),  # Strange random RandomResizedCrop(224,scale = (0.08,0.50)),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.449,), (0.226,))
            ]))
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),  # Strange random RandomResizedCrop(224,scale = (0.08,0.50)),
                transforms.CenterCrop(224),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.449,), (0.226,))
            ])),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

        # transform = transforms.Compose(
        #     [transforms.Grayscale(),transforms.ToTensor(),
        #      transforms.Normalize((0.5,), (0.2,))])

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    if args.start_epoch == 0:  # initiate training
        performance_record = {'epoch': [], 'train_acc': [], 'test_acc': []}
    else:
        performance_record = np.load(record_file_name).item()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc1_train = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        performance_record['epoch'].append(epoch)
        performance_record['train_acc'].append(acc1_train)
        performance_record['test_acc'].append(acc1)

        # remember best acc1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        np.save(record_file_name, performance_record)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'accuracy': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # input = input.cuda(args.gpu, non_blocking=True)
        # target = target.cuda(args.gpu, non_blocking=True)
        input = input.to(device).contiguous()
        target = target.to(device)
        # compute output
        # print(input.dtype)
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
    return top1.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # input = input.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # input = input.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)
            input = input.to(device)
            target = target.to(device)
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best):
    if this_net == "baseline_net":
        filename = 'ImageNet_baseline_checkpoint.pth.tar'
        best_file_name = 'ImageNet_baseline_model_best.pth.tar'
    elif this_net == "SV_net_I":
        filename = 'ImageNet_normalization_checkpoint.pth.tar'
        best_file_name = 'ImageNet_normalization_model_best.pth.tar'
    elif this_net == "SV_net_I_low_frequency":
        filename = 'ImageNet_low_freq_model_checkpoint.pth.tar'
        best_file_name = 'ImageNet_low_freq_model_best.pth.tar'
    elif this_net == "SV_net_II":
        filename = 'ImageNet_SV_net_II_model_checkpoint.pth.tar'
        best_file_name = 'ImageNet_SV_net_II_model_best.pth.tar'
    elif this_net == "SV_net_II_low_frequency":
        filename = 'ImageNet_SV_net_II_low_frequency_model_checkpoint.pth.tar'
        best_file_name = 'ImageNet_SV_net_II_low_frequency_model_best.pth.tar'

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


# this_net “baseline_net”,"SV_net_I","SV_net_I_low_frequency","SV_net_II"

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # TODO: modify output prediction methods:
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()



