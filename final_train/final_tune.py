import sys
# update your projecty root path before running
# sys.path.insert(0, '/home1/lnn/nsga-net-master/')
sys.path.insert(0, '/home1/lnn/nsga-net-master/')
import os
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision


import search.cifar10_search as my_cifar10

import time
from misc import utils

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from misc.flops_counter import add_flops_counting_methods
from supernet.sup_operations_cifar import Imagenet_Models, OPS_containers

# writer = SummaryWriter('runs/cifar10_for_test')
device = 'cuda'

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='model6', help='experiment name')
parser.add_argument('--expr_root', type=str, default='/home1/lnn/nsga-net-master/final_train/' , help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

# ---- train logger ----------------- #
save_pth = os.path.join(args.expr_root, '{}'.format(args.save))
utils.create_exp_dir(save_pth)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_pth, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# def matplotlib_imshow(img, one_channel=False):
#     if one_channel:
#         img = img.mean(dim=0)
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     if one_channel:
#         plt.imshow(npimg, cmap="Greys")
#     else:
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))

def main(model_code, epochs, seed=0, gpu=0, auxiliary=False, cutout=True, drop_path_prob=0.0, ops = None):


    # ---- parameter values setting ----- #
    learning_rate = args.learning_rate
    momentum = args.momentum
    weight_decay = args.weight_decay
    data_root = args.data
    batch_size = args.batch_size
    cutout_length = args.cutout_length
    auxiliary_weight = args.auxiliary_weight
    grad_clip = args.grad_clip
    report_freq = args.report_freq
    train_params = {
        'auxiliary': auxiliary,
        'auxiliary_weight': auxiliary_weight,
        'grad_clip': grad_clip,
        'report_freq': report_freq,
    }




    # logging.info("Genome = %s", genome)
    logging.info("Architecture = %s", model_code)
    OPS_c = ops
    # print(model_code)
    model = Imagenet_Models(input_container=OPS_c, op_code=model_code)

    np.random.seed(seed)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    n_params = (np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, model.parameters())) / 1e6)
    # model = model.to(device)

    model = model.cuda()
    logging.info("param size = %fMB", n_params)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(
        parameters,
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    # CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform, valid_transform = utils._data_transforms_cifar10(args)

    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor()
    # ])

    # if cutout:
    #     train_transform.transforms.append(utils.Cutout(cutout_length))
    #
    # train_transform.transforms.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    # valid_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    # ])

    train_data = my_cifar10.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    valid_data = my_cifar10.CIFAR10(root=data_root, train=False, download=True, transform=valid_transform)

    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    #
    # print('11111111111111111111111111111111111111111111111111111111111111111')
    # dataiter = iter(train_queue)
    # images, labels = dataiter.next()
    #
    # # create grid of images
    # dummy_input = torch.randn(1, 3, 32, 32)
    # with SummaryWriter(comment='model1') as w:
    #     w.add_graph(model, images.cuda())
    # # writer.close()
    # print('222222222222222222222222222222222222222222222222222222222222222222')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))

    # calculate for flops
    model = add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()
    random_data = torch.randn(1, 3, 32, 32)

    model(torch.autograd.Variable(random_data).to(device))

    n_flops = np.round(model.compute_average_flops_cost() / 1e6, 4)
    logging.info('flops = %f', n_flops)

    valid_acc = 0
    inf_time = 0
    best_acc = 0
    for epoch in range(epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = drop_path_prob * epoch / epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, train_params)

        valid_acc, valid_obj, inf_time, = infer(valid_queue, model, criterion)

        if valid_acc > best_acc:
            best_acc = valid_acc
            OPS_c.save_fix_w(path=save_pth + '/')
            utils.save(model, os.path.join(args.save, 'weights.pt'))

            with open(os.path.join(save_pth, 'log.txt'), "w") as file:
                # file.write("Genome = {}\n".format(genome))
                file.write("Architecture = {}\n".format(model_code))
                file.write("param size = {}MB\n".format(n_params))
                file.write("flops = {}MB\n".format(n_flops))
                file.write("valid_acc = {}\n".format(valid_acc))
                file.write("inf_time = {}\n".format(inf_time))

        logging.info('train_acc = %f, valid_acc = %f, inf_time = %f', train_acc, valid_acc, inf_time)



        # logging.info('inf_time %f', inf_time)

    return {
        'valid_acc': valid_acc,
        'params': n_params,
        'flops': n_flops,
        'inf_time': inf_time
    }




# Training
def train(train_queue, net, criterion, optimizer, params):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_queue):
        # inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if params['auxiliary']:
            loss_aux = criterion(targets)
            loss += params['auxiliary_weight'] * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), params['grad_clip'])
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, train_loss/total, 100.*correct/total)
    #
    # logging.info('train acc %f', 100. * correct / total)

    return 100.*correct/total, train_loss/total



def infer(valid_queue, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():
        inf_time = 0
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            start_time = time.time()
            outputs = net(inputs)
            inf_time = time.time() - start_time
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.*correct/total
    # logging.info('valid acc %f', 100. * correct / total)

    return acc, test_loss/total, inf_time*1000


if __name__ == "__main__":

    if args.save == 'modela1':
        code_int = [12, 3, 1, 0, 1, 0, 1, 12, 12, 12, 12, 10, 12, 0, 12, 1, 1]
    elif args.save == 'modela2':
        code_int = [12, 3, 1, 0, 1, 0, 1, 12, 12, 12, 12, 10, 12, 0, 12, 1, 5]
    elif args.save == 'modela3':
        code_int = [12, 3, 1, 0, 0, 0, 1, 9, 12, 12, 12, 11, 10, 0, 7, 1, 1]
    elif args.save == 'modela4':
        code_int = [12, 1, 6, 0, 0, 3, 6, 12, 12, 12, 12, 12, 12, 0, 12, 1, 5]
    elif args.save == 'modela5':
        code_int = [12, 3, 1, 0, 0, 0, 1, 9, 12, 12, 12, 0, 12, 0, 12, 1, 1]
    elif args.save == 'modela6':
        code_int = [12, 3, 1, 0, 0, 0, 1, 12, 12, 12, 12, 0, 12, 0, 12, 1, 1]
    elif args.save == 'modela7':
        code_int = [11, 3, 4, 8, 1, 4, 3, 10, 11, 12, 7, 10, 12, 0, 11, 11, 11]
    elif args.save == 'modela8':
        code_int = [11, 1, 6, 0, 0, 4, 1, 12, 12, 12, 12, 11, 10, 0, 7, 11, 11]
    elif args.save == 'modela9':
        code_int = [11, 3, 6, 0, 1, 4, 1, 10, 11, 12, 7, 11, 7, 0, 7, 11, 12]
    elif args.save == 'modela10':
        code_int = [12, 3, 4, 8, 1, 0, 1, 12, 12, 12, 12, 10, 12, 0, 12, 1, 5]
    elif args.save == 'modelv62':
        code_int = [1, 12, 2, 0, 2, 3, 1, 12, 12, 5, 3, 5, 5, 12, 12, 12, 1]
    elif args.save == 'modelv61':
        code_int = [1, 12, 2, 0, 2, 2, 1, 12, 12, 5, 3, 4, 5, 12, 12, 12, 1]
    elif args.save == 'modelv7':
        code_int = [1, 12, 2, 7, 2, 5, 9, 12, 12, 12, 3, 5, 1, 9, 5, 11, 1]
    elif args.save == 'modelv8':
        code_int = [8, 12, 2, 7, 2, 5, 2, 12, 12, 12, 8, 2, 1, 9, 5, 12, 5]
    elif args.save == 'modelv9':
        code_int = [1, 12, 2, 12, 1, 5, 2, 12, 0, 12, 3, 5, 5, 12, 5, 12, 0]
    elif args.save == 'modelv10':
        code_int = [11, 3, 11, 4, 2, 7, 6, 12, 10, 12, 9, 2, 4, 1, 5, 11, 10]
    else:
        # code_int = [1, 12, 2, 0, 2, 3, 1, 12, 12, 5, 3, 5, 5, 12, 12, 12, 1]
        code_int = [7, 10, 5, 6, 2, 5, 10, 4, 7, 5, 0, 4]


    start = time.time()
    print(main(model_code=code_int, epochs=args.epochs, seed=1,
               auxiliary=False, cutout=True, drop_path_prob=0.2, ops=OPS_containers(init_c=args.init_channels, path=save_pth + '/')))
    print('Time elapsed = {} mins'.format((time.time() - start)/60))

