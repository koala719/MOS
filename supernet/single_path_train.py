from __future__ import print_function, division
import sys
# update your projecty root path before running
sys.path.insert(0, '/home1/lnn/nsga-net-master')

# import sys
import argparse
import os
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import models
import time

from misc import utils
# from supernet import

from supernet.accuracy import accuracy
from supernet.dataloader import get_imagenet_dataset
from supernet.sup_operations_cifar import Imagenet_Models, OPS_containers
from misc.flops_counter import add_flops_counting_methods
import search.cifar10_search as my_cifar10
import inspect
# from supernet.gpu_mem_track import MemTracker



parser = argparse.ArgumentParser(description='Single_Path Config')
# parser.add_argument('--model', default='Scarlet_A', choices=['Scarlet_A', 'Scarlet_B', 'Scarlet_C'])
parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu'])
parser.add_argument('--val-dataset-root', default='/home/work/dataset/ILSVRC2012', type=str, help="val dataset root path")
parser.add_argument('--save_path', default='./', type = str, help="the path  of output")
parser.add_argument('--weight_path', default='./output/', type = str, help="the path  of output")
parser.add_argument('--batch_size', default=512 , type=int, help='trn batch size')
parser.add_argument('--gpu_id', default=0, type=int, help='gpu to run')
# parser.add_argument('--class_num', default=None, type=int, help='number of class')
parser.add_argument('--lr_rate', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=3e-4, type=float, help='weight_decay')
parser.add_argument('--data_path', default='./data', type=str, help='the path of data')
parser.add_argument('--cutout_len', default=16, type=int, help='the length of cutout')
parser.add_argument('--aux_weight', default=0.4, type=float, help='auxiliary weight')
parser.add_argument('--auxiliary', default=False, type=bool, help='is auxiliary')
parser.add_argument('--grad_clip', default=6, type=int, help='grad clip')
parser.add_argument('--report_freq', default=50, type=int, help='the frequence of report')
parser.add_argument('--dataset', default='cifar', type=str, help='cifar or imagenet')
parser.add_argument('--layers', default=18, type=int, help='layers of model')
parser.add_argument('--model_num', default=13, type=int, help='number of model')
parser.add_argument('--seed', default=0, type=int, help='manual seed')
parser.add_argument('--epochs', default=2, type=int, help='train epochs')
parser.add_argument('--drop_path_prob', default=0.0, type=float, help='drop path prob')
parser.add_argument('--ops', default=13, type=int, help='number of operations')
parser.add_argument('--cutout', default=True, type=bool, help='is cutout')
parser.add_argument('--iters', default=1000, type=int, help='train iterations')

args = parser.parse_args()
# print(args)
args.save_path = os.path.join(args.save_path, 'output')
utils.create_exp_dir(args.save_path)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def model_sampling(layers, model_num, ops):
    # model_list = []

    model_op = []
    for i in range(layers):
        op_list = np.random.choice(ops, model_num, replace=False)
        model_op.append(op_list)
    model_op = np.array(model_op)
    model_list = model_op.T

    return model_list


def main():
    train_params = {
        'auxiliary': args.auxiliary,
        'auxiliary_weight': args.aux_weight,
        'grad_clip': args.grad_clip,
        'report_freq': args.report_freq,
    }
    OPS_c = OPS_containers(path=args.weight_path)

    layers = args.layers
    model_num = args.model_num
    ops = args.ops
    # Model = {}
    parameter_total = nn.ParameterList()
    for i in range(13):
        for j in range(18):
            for param in OPS_c.OPS[j][i].parameters():
                parameter_total.append(param)

    t_params = (np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, parameter_total)) / 1e6)

    print('total_params:',t_params)


    ###### data
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    if args.cutout:
        train_transform.transforms.append(utils.Cutout(args.cutout_len))

    train_transform.transforms.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_data = my_cifar10.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
    valid_data = my_cifar10.CIFAR10(root=args.data_path, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,shuffle=True,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,shuffle=False,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    optimizer = torch.optim.SGD(
        parameter_total,
        args.lr_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    torch.cuda.set_device(args.gpu_id)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    # device = torch.device('cuda:0')

    # frame = inspect.currentframe()
    # gpu_tracker = MemTracker(frame)  # 创建显存检测对象
    #
    # gpu_tracker.track()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(args.iters))
    # total_step = 1
    for iter in range(args.iters):

        # Train_loss = []
        # Total = []
        # Correct = []
        scheduler.step()
        for step, (inputs, targets) in enumerate(train_queue):

            inputs, targets = inputs.cuda(), targets.cuda()

            model_list = model_sampling(layers, model_num, ops)
            model = {}
            for i in range(model_num):
                model[i] = Imagenet_Models(input_container=OPS_c, op_code=model_list[i])

            # Loss = []
            Outputs = []
            # Predicted = []

            train_loss_tot = 0
            # correct = 0
            # total = 0


            # for epoch in range(args.epochs):

            try:
                for i in range(13):
                    m_acc = model[i]
                    # m_acc = m_acc.cuda()
                    m_acc = nn.DataParallel(m_acc).cuda()

                    m_acc.droprate = args.drop_path_prob * iter / args.iters
                    ########################################################################
                    optimizer.zero_grad()
                    ########################################################################
                    # gpu_tracker.track()
                    outputs = m_acc(inputs)
                    # gpu_tracker.track()
                    ########################################################################

                    _, predicted_t = outputs.max(1)
                    # gpu_tracker.track()

                    # print('lidafanshigedahuaidan train error:', predicted_t.eq(targets).sum().item())
                    loss = criterion(outputs, targets)
                    # gpu_tracker.track()
                    train_loss_tot = train_loss_tot + loss
                    # gpu_tracker.track()
                    # Loss.append(loss)
                    Outputs.append(outputs)
                    # gpu_tracker.track()


                    # loss.backward()
                # gpu_tracker.track()
                train_loss_tot.backward()
                # gpu_tracker.track()

                nn.utils.clip_grad_norm_(parameter_total, train_params['grad_clip'])
                optimizer.step()
                # gpu_tracker.track()


                for i_loss in range(len(Outputs)):
                    _, predicted = Outputs[i_loss].max(1)
                    # Predicted.append(predicted)
                    C_each = predicted.eq(targets).sum().item()
                    # Correct.append(C_each)

                    train_acc, train_obj = 100. * C_each / targets.size(0), train_loss_tot
                    if step % 100 == 0:
                        logging.info('epoch %d step %d model %d lr %e train_loss_tot = %f train_acc %f', iter, step,
                                     i_loss,
                                     scheduler.get_lr()[0], train_loss_tot, train_acc)

            except Exception as e:
                print(e)


            # for i in range(model_num):
            #     del model[i]
            # del model

        # if total_step % 100==0:
        # if step %50==0:
        ###########################################################################
        for m_i in range(len(model_list)):
            parameter_list = nn.ParameterList()
            for j in range(len(model_list[m_i])):
                for param in OPS_c.OPS[j][model_list[m_i][j]].parameters():
                    parameter_list.append(param)
            n_params = (np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, parameter_list))/ 1e6)

            m = model[m_i]
            m = m.cuda()
            # logging.info("param size = %fMB", n_params)
            m = add_flops_counting_methods(m)
            m.eval()
            m.start_flops_count()
            random_data = torch.randn(1, 3, 32, 32)
            m(torch.autograd.Variable(random_data).to(args.device))
            n_flops = np.round(m.compute_average_flops_cost() / 1e6, 4)
            # logging.info('flops = %f', n_flops)

            # train_acc, train_obj = 100. * C_each / targets.size(0), train_loss_tot# Train_loss / Total
            # train_acc, train_obj = train(inputs, targets, m, criterion, optimizer, train_params)
            # logging.info('train_acc %f', train_acc)
            valid_acc, valid_obj = infer(valid_queue, m, criterion)


            logging.info("param size = %fMB, flops = %f, train_acc %f, valid_acc %f",
                         n_params,
                         n_flops,
                        train_acc,
                        valid_acc)

            with open(os.path.join(args.save_path, 'log.txt'), "w") as file:
                file.write("Genome = {}\n".format(model_list[m_i]))
                file.write("Architecture = {}\n".format(m))
                file.write("param size = {}MB\n".format(n_params))
                file.write("flops = {}MB\n".format(n_flops))
                file.write("valid_acc = {}\n".format(valid_acc))
                # file.write("dict = {}\n".format(m.state_dict()))
        # if total_step % 5000 == 0 and step >0:
        OPS_c.save_fix_w(path=args.weight_path)

    return {
        'valid_acc': valid_acc,
        'params': n_params,
        'flops': n_flops,
    }


def train(train_queue, net, criterion, optimizer, params):
    net.train()
    train_loss = 0
    correct = 0
    total = 0


    for step, (inputs, targets) in enumerate(train_queue):
        # inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.cuda(), targets.cuda()

        # outputs, outputs_aux = net(inputs)

        # with torch.no_grad():
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print("before update",correct/total*100.)
        # if params['auxiliary']:
        #     loss_aux = criterion(outputs_aux, targets)
        #     loss += params['auxiliary_weight'] * loss_aux
        # g = loss.grad()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), params['grad_clip'])
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print("after update", correct / total * 100.)



    return 100.*correct/total, train_loss/total, loss

def infer(valid_queue, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            # outputs, _ = net(inputs)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # if step % args.report_freq == 0:
            #     logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.*correct/total
    # logging.info('valid acc %f', 100. * correct / total)

    return acc, test_loss/total


if __name__ == "__main__":
    model_list = model_sampling(args.layers, args.model_num, args.ops)
    start = time.time()
    print(main())
    print('Time elapsed = {} mins'.format((time.time() - start)/60))

