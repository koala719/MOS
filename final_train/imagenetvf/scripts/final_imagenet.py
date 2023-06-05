import sys
# update your projecty root path before running
sys.path.insert(0, '/home/drl/lnn/nsga-net-master/')

import os
import sys
import numpy as np
import time
import torch
from final_train import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
# import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from misc.flops_counter import add_flops_counting_methods
from supernet.sup_operations_imagenet import Imagenet_Models, OPS_containers
# from model import NetworkImageNet as Network

# code_int = [10, 3, 11, 0, 2, 2, 4, 6, 0, 2, 9, 0, 6, 5, 6, 4, 8, 10, 3, 11, 0, 2, 2, 4, 6, 0, 2, 9, 0, 6, 5, 6, 4, 8]
# code_int = [10, 3, 11, 11, 0, 2, 2, 4, 6, 0, 2, 9, 0, 6, 6, 10, 3, 11, 11, 0, 2, 2, 4, 6, 0, 2, 9, 0, 6, 6]
code_int = [10, 10, 3, 3, 11, 11, 11, 11, 0, 0, 2, 2, 2, 2, 4, 4, 6, 6, 0, 0, 2, 2, 9, 9, 0, 0, 6, 6, 5, 5, 6, 6, 4, 4, 8, 8]

# code_int = [8, 5, 10, 11, 3, 11, 4, 2, 9, 9, 8, 10, 2, 4, 6, 8, 5, 10, 11, 3, 11, 4, 2, 9, 9, 8, 10, 2, 4, 6]
# code_int = [1, 12, 2, 0, 2, 3, 1, 12, 12, 5, 3, 5, 5, 12, 12, 12, 1]0
parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='/home1/dzx/darts-master/data/imagenet/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=36, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='./imagenetvf/', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=30, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=True, help='data parallelism')
parser.add_argument('--resume', default='./imagenetvf', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--tencrop', action='store_true', default=True, help='use 10-crop testing')
parser.add_argument('--mode', type=str, default='train', help='experiment mode: train or test')
parser.add_argument('--num_readers', type=int, default=16, help='total number of layers')
parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--reset_save_dir', action='store_true', default=False, help='')

args = parser.parse_args()

# args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  ops = OPS_containers(init_c=args.init_channels, path=args.save)
  model_code = code_int
  logging.info("Architecture = %s", model_code)
  OPS_c = ops
  # print(model_code)
  model = Imagenet_Models(input_container=OPS_c, op_code=model_code)
  if args.parallel:
    model = nn.DataParallel(model).cuda()
  else:
    model = model.cuda()


  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


  if args.resume:
    if args.mode == 'train':
      args.resume = os.path.join(args.resume, 'checkpoint.pth.tar')
    elif args.mode == 'train1':
      args.resume = os.path.join(args.resume, 'model_best.pth.tar')
    else:
      raise ValueError("Unknown mode '{0}'".format(args.mode))

    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)
      args.start_epoch = checkpoint['epoch']
      best_acc_top1 = checkpoint['best_acc_top1']
      model.load_state_dict(checkpoint['state_dict'])
      print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )

  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)

  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 150, 180], gamma=args.gamma)
  # start_epochs = 174
  best_acc_top1 = 0
  for epoch in range(args.start_epoch, args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc_top1 %f', valid_acc_top1)
    logging.info('valid_acc_top5 %f', valid_acc_top5)

    is_best = False
    if valid_acc_top1 > best_acc_top1:
      best_acc_top1 = valid_acc_top1
      is_best = True

    utils.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc_top1': best_acc_top1,
      'optimizer' : optimizer.state_dict(),
      }, is_best, args.save)
    if is_best:
        OPS_c.save_fix_w(path=args.save + '/')
        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()
  for step, (input, target) in enumerate(train_queue):
    target = target.cuda(async=True)
    input = input.cuda()
    input = Variable(input)
    target = Variable(target)

    optimizer.zero_grad()


    logits = model(input)
    loss = criterion(logits, target)
    # if args.auxiliary:
    #   loss_aux = criterion(logits_aux, target)
    #   loss += args.auxiliary_weight*loss_aux

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    # objs.update(loss.data[0], n)
    # top1.update(prec1.data[0], n)
    # top5.update(prec5.data[0], n)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
      for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        # objs.update(loss.data[0], n)
        # top1.update(prec1.data[0], n)
        # top5.update(prec5.data[0], n)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main()
