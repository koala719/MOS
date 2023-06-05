import math
import torch.nn as nn
from supernet.basic_ops import *
import random
import numpy as np
#
# OPS = {}
# OPS_layer = {}
# layers = 17
# ops = 13
class OPS_containers():
    def __init__(self, n_class = 10, path = None):
        OPS = {}

        layers = 17
        ops = 13
        for i in range(layers):
            OPS_layers = { 0: lambda C_in, C_out, stride, affine: InvertedResidual(C_in, C_out, 3, stride, expand_ratio=3, is_use_se=False),
                           1: lambda C_in, C_out, stride, affine: InvertedResidual(C_in, C_out, 3, stride, expand_ratio=3, is_use_se=True),
                           2: lambda C_in, C_out, stride, affine: InvertedResidual(C_in, C_out, 5, stride, expand_ratio=3, is_use_se=False),
                           3: lambda C_in, C_out, stride, affine: InvertedResidual(C_in, C_out, 5, stride, expand_ratio=3, is_use_se=True),
                           4: lambda C_in, C_out, stride, affine: InvertedResidual(C_in, C_out, 7, stride, expand_ratio=3, is_use_se=False),
                           5: lambda C_in, C_out, stride, affine: InvertedResidual(C_in, C_out, 7, stride, expand_ratio=3, is_use_se=True),
                           6: lambda C_in, C_out, stride, affine: InvertedResidual(C_in, C_out, 3, stride, expand_ratio=6, is_use_se=False),
                           7: lambda C_in, C_out, stride, affine: InvertedResidual(C_in, C_out, 3, stride, expand_ratio=6, is_use_se=True),
                           8: lambda C_in, C_out, stride, affine: InvertedResidual(C_in, C_out, 5, stride, expand_ratio=6, is_use_se=False),
                           9: lambda C_in, C_out, stride, affine: InvertedResidual(C_in, C_out, 5, stride, expand_ratio=6, is_use_se=True),
                           10: lambda C_in, C_out, stride, affine: InvertedResidual(C_in, C_out, 7, stride, expand_ratio=6, is_use_se=False),
                           11: lambda C_in, C_out, stride, affine: InvertedResidual(C_in, C_out, 7, stride, expand_ratio=6, is_use_se=True),
                           12: lambda C_in, C_out, stride, affine: nn.Conv2d(C_in, C_out, 1, padding=0, bias=False)
                          }
            OPS_layer = {}
            for j in range(ops):
                if i == 0:
                    OPS_layer[j] = OPS_layers[j](16, 32, 2, True)
                elif i == 1:
                    OPS_layer[j] = OPS_layers[j](32, 32, 1, True)
                elif i == 2:
                    OPS_layer[j] = OPS_layers[j](32, 40, 2, True)
                elif i in [3, 4, 5]:
                    OPS_layer[j] = OPS_layers[j](40, 40, 1, True)
                elif i == 6:
                    OPS_layer[j] = OPS_layers[j](40, 80, 2, True)
                elif i in [7, 8, 9]:
                    OPS_layer[j] = OPS_layers[j](80, 80, 1, True)
                elif i == 10:
                    OPS_layer[j] = OPS_layers[j](80, 96, 1, True)
                elif i in [11, 12, 13]:
                    OPS_layer[j] = OPS_layers[j](96, 96, 1, True)
                elif i == 14:
                    OPS_layer[j] = OPS_layers[j](96, 192, 2, True)
                elif i in [15, 16, 17]:
                    OPS_layer[j] = OPS_layers[j](192, 192, 1, True)
                else:
                    OPS_layer[j] = OPS_layers[j](192, 320, 1, True)
            OPS[i] = OPS_layer
        self.OPS = OPS
        last_channel = 320

        self.last_channel = last_channel
        self.stem = stem(3, 32, 2)
        self.separable_conv = separable_conv(32, 16)

        self.conv_before_pooling = conv_before_pooling(320, self.last_channel)
        self.classifier = nn.Linear(self.last_channel, n_class)
        try:
            self.load_fix_w(path)
            print('sucess!!!!')
        except:
            print("Fail")
            pass
        # except Exception as e:
        #     print(e)
        #     print('Fail!!!!')





    def save_fix_w(self, path):
        torch.save(self.stem.state_dict(), path + 'stem.pkl')
        torch.save(self.separable_conv.state_dict(), path + 'sep_conv.pkl')
        torch.save(self.conv_before_pooling.state_dict(), path + 'conv_before_pooling.pkl')
        torch.save(self.classifier.state_dict(), path + 'classifier.pkl')
        for i in range(19):
            for j in range(12):
                self.OPS[i][j].save_weight(path + 'mb_module' + str(i) + '_' + str(j) + '.pkl')
            torch.save(self.OPS[i][12].state_dict(), path + 'mb_module' + str(i) + '_' + '12' + '.pkl')


    def load_fix_w(self, path):
        self.stem.load_state_dict(torch.load(path + 'stem.pkl'))
        self.separable_conv.load_state_dict(torch.load(path + 'sep_conv.pkl'))
        self.conv_before_pooling.load_state_dict(torch.load(path + 'conv_before_pooling.pkl'))
        self.classifier.load_state_dict(torch.load(path + 'classifier.pkl'))
        for i in range(17):
            for j in range(12):
                self.OPS[i][j].load_weight(path + 'mb_module' + str(i) + '_' + str(j) + '.pkl')
            self.OPS[i][12].load_state_dict(torch.load(path + 'mb_module' + str(i) + '_' + '12' + '.pkl'))
        # print(self.OPS)


class Imagenet_Models(nn.Module):

    def __init__(self, n_class = 10, input_size = 32, input_container=None, op_code=None):
        super(Imagenet_Models, self).__init__()
        assert input_size % 32 == 0

        input_channel = 16
        last_channel = 192
        OPS = input_container.OPS
        self.stem = input_container.stem
        self.last_channel = input_container.last_channel
        self.separable_conv = input_container.separable_conv
        # self.stem = stem(3, 32, 2)
        # self.separable_conv = separable_conv(32, 16)
        self.mb_module = list()
        # print (op_code)
        for i in range(len(op_code)):
            self.mb_module.append(OPS[i][op_code[i]])
        # print (self.mb_module)

        self.mb_module = nn.Sequential(*self.mb_module)
        # self.conv_before_pooling = conv_before_pooling(192, self.last_channel)
        self.conv_before_pooling = input_container.conv_before_pooling
        self.classifier = input_container.classifier
        # self.classifier = nn.Linear(self.last_channel, n_class)
        # self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.separable_conv(x)
        x = self.mb_module(x)
        x = self.conv_before_pooling(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

