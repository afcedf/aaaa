import os
import argparse
import time
from datetime import timedelta
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from torchvision.models import resnet152

from data_transforms import Transforms
from dataset import TwoStreamSet


parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, choices=['RGB', 'FLOW'])

parser.add_argument('--trainlist', type=str, default='ucfTrainTestlist/trainlist01.txt')

parser.add_argument('--vallist', type=str, default='ucfTrainTestlist/testlist01.txt')

parser.add_argument('--num_classes', type=int, default=101)

parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument('--weight_decay', type=float, default=0.0005)

parser.add_argument('--start_epoch', type=int, default=0)

parser.add_argument('--resume', type=str, default=None)

best_prec = 0.
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

def main():

    global best_prec, args
    args = parser.parse_args()

    model = resnet152(pretrained=True)

    if args.mode == 'FLOW':
        first_conv_weight = model.conv1.weight.data.mean(dim=1, keepdim=True)
        model.conv1 = nn.Conv2d(20, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.conv1.weight.data.copy_(first_conv_weight, broadcast=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, args.num_classes)

    model = nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            ckpt = torch.load(args.resume)
            args.start_epoch = ckpt['epoch']
            best_prec = ckpt['best_prec']
            model.load_state_dict(ckpt['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    cudnn.benchmark = True

    DataSet = {
        'train': TwoStreamSet(args.trainlist, mode=args.mode, phase='train', transforms=Transforms['train'][args.mode]),
        'val': TwoStreamSet(args.vallist, mode=args.mode, phase='val', transforms=Transforms['val'][args.mode])
    }

    train_loader = torch.utils.data.DataLoader(
        DataSet['train'], batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        DataSet['val'], batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.mode == 'RGB':
        init_lr = 0.001
        epochs = 20
        optimizer = optim.SGD(model.parameters(), init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[6 - args.start_epoch], gamma=0.1)

    elif args.mode == 'FLOW':
        init_lr = 0.005
        epochs = 35
        optimizer = optim.SGD(model.parameters(), init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5 - args.start_epoch, 20 - args.start_epoch], gamma=0.1)



    for epoch in range(args.start_epoch, epochs):
        scheduler.step()

        print '-' * 30 + 'epoch: %d' % (epoch + 1) + '-' * 30
        print 'learning rate: %f' % optimizer.param_groups[0]['lr']

        print 'Training...'
        start_time = time.time()
        loss, top1 = train(train_loader, model, criterion, optimizer)
        print 'loss: %f, top1 accuracy: %.2f%%, time: %s' % (loss, top1, str(timedelta(seconds=time.time()-start_time))[:-7])

        print 'Validating...'
        start_time = time.time()
        loss, top1 = validate(val_loader, model, criterion)
        print 'loss: %f, top1 accuracy: %.2f%%, time: %s' % (loss, top1, str(timedelta(seconds=time.time()-start_time))[:-7])

        is_best = top1 > best_prec
        best_prec = max(top1, best_prec)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
        }, args.mode, is_best)

def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    for i, (input_, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input_var = Variable(input_)
        target_var = Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec, = accuracy(output.data, target, topk=(1,))

        losses.update(loss.data[0], input_.size(0))
        top1.update(prec[0], input_.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg

def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    for i, (input_, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = Variable(input_, volatile=True)
        target_var = Variable(target, volatile=True)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec, = accuracy(output.data, target, topk=(1,))

        losses.update(loss.data[0], input_.size(0))
        top1.update(prec[0], input_.size(0))

    return losses.avg, top1.avg

class AverageMeter(object):

    def __init__(self):
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

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def save_checkpoint(state, mode, is_best, prefix='./weights/two_stream_'):
    filename = prefix + mode + '_latest2.pth.tar'
    torch.save(state, filename)

    if is_best:
        bestfilename = prefix + mode + '_best2.pth.tar'
        shutil.copyfile(filename, bestfilename)

if __name__ == '__main__':
    main()