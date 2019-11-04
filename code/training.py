# training function that merges previous repeated codes

import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import model as models
from datetime import date

# TODO: Adjust to be compatbale with MNIST and CIFAR training
# TODO: concatinate this list of models, with better explanation for each of them.

this_net = ["Vanilla_ResNet", "SV_Net_I", "SV_Net_I_Low_Freq", "SV_Net_II", "SV_Net_II_Low_Freq", "SV_Net_LN",
            "SV_Net_All_Freq"]

Data = ["ImageNet128","ImageNet224", "CIFAR10", "MNIST"]
# this_net “Vanilla_ResNet”,"SV_net_I","SV_net_I_low_frequency","SV_net_II"

# TODO: test code with training and evaluation
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# TODO: add this net to choice one of the arguments
parser = argparse.ArgumentParser(description='Set up training parameters')

parser.add_argument('-path', metavar='DIR', default='/gpfs01/bethge/data/imagenet-raw/raw-data',
                    help='path to data set, default is the imagenet location')

parser.add_argument('-d', '--data', metavar='DATA', default='ImageNet128', dest="data",
                    choices=Data, help='Data Used to Train: ' + ' | '.join(Data) + ' (default: ImageNet128)')

# TODO: this_net should not be a list.
parser.add_argument('-a', '--arch', metavar='ARCH', default='Vanilla_ResNet', dest="arch",
                    choices=this_net, help='model architecture: ' + ' | '.join(this_net) + ' (default: resnet18)')
parser.add_argument('--epochs', default=80, type=int, metavar='N', dest="epochs",
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', dest="start_epoch",
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, dest="batchsize",
                    metavar='N',
                    help='mini-batch size (default: 4), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', dest="momentum",
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int, dest= "seed",
                    help='seed for initializing training. ')


best_acc1 = 0


def main():
    """Collect User Defined Variables"""

    args = parser.parse_args()

    # Namespace(arch='Vanilla_ResNet', batch_size=256, epochs=80, evaluate=False, lr=0.1,
    # momentum=0.9, path='\\data',
    # print_freq=10, resume='', seed=None, start_epoch=0, weight_decay=0.0001)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    global best_acc1

    # define the model used for training.
    # TODO: implement integrated net across 12 different frequencies.
    print("=> creating model '{}'".format(args.arch))
    print('batch size is ', args.batchsize)
    if args.data == "ImageNet128":
        image_size = 128
        n_classes = 1000
    elif args.data == "ImageNet224":
        image_size = 224
        n_classes = 1000
    elif args.data == "CIFAR10":
        image_size = 32
        n_classes = 10
    elif args.data == "MNIST":
        image_size = 32
        n_classes = 10
    else:
        print("No such data set!")

    filename = args.arch + '_checkpoint.pth.tar'
    best_file_name = args.arch + '_model_best.pth.tar'
    if args.arch == "Vanilla_ResNet":  # make sure the simple net is working for pilot training.
        # TODO: IF WORKS
        print('Employing Vanilla ResNet')
        model = models.construct_resnet18(num_classes=n_classes)
    elif args.arch == "SV_Net_I":
        # TODO: IF WORKS

        print('Employing the first version of Spatial Vision Net')
        model = models.construct_svnet_1(imsize=image_size, num_classes=n_classes)
    elif args.arch == "SV_Net_I_Low_Freq":
        # TODO: IF WORKS

        print('Employing the low frequency version of Spatial Vision Net')
        model = models.construct_svnet_low_f(imsize=image_size, num_classes=n_classes)
    elif args.arch == "SV_Net_II":  # make sure the simplenet is working for pilot training.
        print('Employing simple net')
        # TODO: IF WORKS

        model = models.construct_svnet_2(imsize=image_size, num_classes=n_classes)
    elif args.arch == "SV_Net_II_Low_Freq":
        # TODO: IF WORKS

        print('Employing simple net with low frequency')
        model = models.construct_svnet_2_low_f(imsize=image_size,num_classes=n_classes)
    elif args.arch == "SV_Net_LN":
        # TODO: IF WORKS

        model = models.construct_integrated_net(imsize=image_size,num_classes=n_classes)
    elif args.arch == "SV_Net_All_Freq":
        # TODO: IF WORKS

        print('Employing simple net with low frequency')
        model = models.construct_SV_net_all_freq(imsize=image_size, num_classes=n_classes)

    today = date.today()
    this_net = ["Vanilla_ResNet", "SV_Net_I", "SV_Net_I_Low_Freq", "SV_Net_II", "SV_Net_II_Low_Freq", "SV_Net_LN",
                "SV_Net_All_Freq"]
    Data = ["ImageNet128", "ImageNet224", "CIFAR10", "MNIST"]
    #
    # record_file_name = args.data + '_Performance_Record_' + args.arch + "_" + args.epochs + args.start_epoch + args.batchsize + args.lr + args.momentum + args.weight_decay + args.seed, str(
    #     today) + '.npy'  # File to record learning rate

    record_file_name = args.data + '_Performance_Record_' + args.arch + "_" + str(today) + '.npy'  # File to record learning rate

    # this_net “baseline_net”,"SV_net_I","SV_net_I_low_frequency","SV_net_II"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model).to(device)
    # TODO: modify loss criterion to be compatible with independent filter responses
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # TODO: check optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['accuracy']
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    #
    # train_dir = os.path.join(args.path, 'train')
    # val_dir = os.path.join(args.path, 'val')

    train_dir = args.path
    val_dir = args.path
    # normalization for RGB values
    if args.data == "CIFAR10":
        normalize_rgb = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        normalize_gray = transforms.Normalize((0.5,), (0.5,))
        # transforms sequentially, mean and variance prespecified
    elif args.data == "MNIST":
        normalize_rgb = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        normalize_gray = transforms.Normalize((0.5,), (0.5,))
    else:
        normalize_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize_gray = transforms.Normalize((0.449,), (0.226,))

    # data pre processing according to network type
    if this_net == "baseline_net":
        train_data = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_rgb,
            ]))
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
                normalize_rgb,
            ])), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    else:  # For networks that are not vanilla, use gray images for training.
        train_data = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(image_size),  # Strange random RandomResizedCrop(224,scale = (0.08,0.50)),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(),
                transforms.ToTensor(),
                normalize_gray,
            ]))
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
                normalize_gray,
            ])),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.start_epoch == 0:  # initiate training
        # TODO: make performance record better.
        performance_record = {'epoch': [], 'train_acc': [], 'test_acc': []}
    else:
        performance_record = np.load(record_file_name).item()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc1_train = train(train_loader, model, criterion, optimizer, epoch, args)

        # at the end of each epoch, evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # update performance record
        performance_record['epoch'].append(int(epoch))
        performance_record['train_acc'].append(float(acc1_train))
        performance_record['test_acc'].append(float(acc1))

        # remember best acc1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        np.save(record_file_name, performance_record)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'accuracy': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best,filename,best_file_name)


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
    """returns top 1 average of model tested on validation set"""

    # TODO: check out how does this AverageMeter work
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
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            input = input.to(device)
            target = target.to(device)
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


def save_checkpoint(state, is_best, filename, best_file_name):
    # TODO: add additional save functions here.
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


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
    # TODO: what does this function do?

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        # TODO: what is self.__dict__ ?
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    # TODO: what does *meters mean?
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



