from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F 
import torch.utils.model_zoo as model_zoo

from .base_model import BaseModel, BaseModelPlanA, BaseModelPlanA_Three
import sgtapose
import copy
from torch import batch_norm, einsum
from einops import rearrange, repeat
try:
    from .DCNv2.dcn_v2 import DCN
except:
    print('import DCN failed')
    DCN = None


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False,
                 opt=None):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)
        if opt.pre_img:
            self.pre_img_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        if opt.pre_hm:
            self.pre_hm_layer = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=7, stride=1,
                    padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        if opt.ct_modify:
            self.repro_hm_layer = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=7, stride=1,
                    padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x=None, pre_img=None, pre_hm=None, repro_hm=None):
        y = []
        if x is not None:
            x = self.base_layer(x)
            if pre_img is not None:
                x = x + self.pre_img_layer(pre_img)
            if pre_hm is not None:
                x = x + self.pre_hm_layer(pre_hm)
            if repro_hm is not None:
                x=  x + self.repro_hm_layer(repro_hm)
        else:
            if pre_img is not None:
                x = self.pre_img_layer(pre_img)
                if pre_hm is not None:
                    x = x + self.pre_hm_layer(pre_hm)
            else:
                if pre_hm is not None:
                    x = self.pre_hm_layer(pre_hm)
                  
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights, strict=False)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(
            data='imagenet', name='dla34', hash='ba72cf86')
    else:
        print('Warning: No ImageNet pretrain!!')
    return model
    
def dla341(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(
            data='imagenet', name='dla34', hash='ba72cf86')
    else:
        print('Warning: No ImageNet pretrain!!')
    return model

def dla102(pretrained=None, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(
            data='imagenet', name='dla102', hash='d94d9790')
    return model

def dla46_c(pretrained=None, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=Bottleneck, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla46_c', hash='2bfd52c3')
    return model


def dla46x_c(pretrained=None, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla46x_c', hash='d761bae7')
    return model


def dla60x_c(pretrained=None, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla60x_c', hash='b870c45c')
    return model


def dla60(pretrained=None, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla60', hash='24839fc4')
    return model


def dla60x(pretrained=None, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla60x', hash='d15cacda')
    return model


def dla102x(pretrained=None, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla102x', hash='ad62be81')
    return model


def dla102x2(pretrained=None, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla102x2', hash='262837b6')
    return model


def dla169(pretrained=None, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla169', hash='0914e092')
    return model


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class Conv(nn.Module):
    def __init__(self, chi, cho):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.conv(x)


class GlobalConv(nn.Module):
    def __init__(self, chi, cho, k=7, d=1):
        super(GlobalConv, self).__init__()
        gcl = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=(k, 1), stride=1, bias=False, 
                                dilation=d, padding=(d * (k // 2), 0)),
            nn.Conv2d(cho, cho, kernel_size=(1, k), stride=1, bias=False, 
                                dilation=d, padding=(0, d * (k // 2))))
        gcr = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=(1, k), stride=1, bias=False, 
                                dilation=d, padding=(0, d * (k // 2))),
            nn.Conv2d(cho, cho, kernel_size=(k, 1), stride=1, bias=False, 
                                dilation=d, padding=(d * (k // 2), 0)))
        fill_fc_weights(gcl)
        fill_fc_weights(gcr)
        self.gcl = gcl
        self.gcr = gcr
        self.act = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.gcl(x) + self.gcr(x)
        x = self.act(x)
        return x


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f, node_type=(DeformConv, DeformConv)):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = node_type[0](c, o)
            node = node_type[1](o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None, 
                 node_type=DeformConv):
        super(DLAUp, self).__init__()
        self.startp = startp # startp: 2, channels: [64, 128, 256, 512], sclaes = [1,2,4,8]
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1): # len(channels) == 4
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j],
                          node_type=node_type))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


DLA_NODE = {
    'dcn': (DeformConv, DeformConv),
    'gcn': (Conv, GlobalConv),
    'conv': (Conv, Conv),
}

class DLASeg(BaseModel):
    def __init__(self, num_layers, heads, head_convs, opt):
        super(DLASeg, self).__init__(
            heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        down_ratio=4
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]
        # print('Using node type:', self.node_type)
        # print('this one')
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        self.base = globals()['dla{}'.format(num_layers)](
            pretrained=(opt.load_model == ''), opt=opt)
        
        # print(self.base)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:], scales,
            node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level], 
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type)
        

    def img2feats(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]

    def imgpre2feats(self, x, pre_img=None, pre_hm=None, repro_hm=None):
        x = self.base(x, pre_img, pre_hm,repro_hm)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: BxCxHxW
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

def Two_Frames_Fusion(cur_features, prev_features, opt):
    B, C, _, _ = cur_features.shape
    spatial_attn = SpatialAttention(kernel_size=3).to(opt.device)
    prev_features = prev_features * spatial_attn(cur_features)
    x = torch.cat([cur_features, prev_features], dim = 1)
    conv = nn.Sequential(nn.Conv2d(2 * C, C, 3, padding=1, bias=True), nn.ReLU(inplace=True)).to(opt.device)
    return conv(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_inp, d_model, n_k, d_ffn=1024,
                 dropout=0.1,
                 n_heads=8, pos_embed=True):
        # d_out = d_model * n_heads
        super().__init__()
        self.d_model = d_model
        self.d_inp = d_inp
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.n_heads = n_heads
        self.d_out = self.d_model * self.n_heads
        
        # cross attention
        self.cross_attn = MHCA_ein(self.n_heads, self.d_inp, self.d_out, n_k, pos_embed=pos_embed)
        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.d_inp)
        
        # ffn
        self.linear1 = nn.Linear(self.d_inp, self.d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.d_ffn, self.d_inp)
        self.dropout4 = nn.Dropout(self.dropout)
        self.norm3 = nn.LayerNorm(self.d_inp)
    
    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward(self, query, key, value):
        # cross-attention
        tgt = self.cross_attn(query, key, value)
        query = tgt + self.dropout1(query)
        query = self.norm1(query)
        
        # ffn
        query = self.forward_ffn(query)
        
        return query
        
class TransformerEncoderLayerOri(nn.Module):
    def __init__(self, d_inp, d_model, d_ffn=1024,
                 dropout=0.1,
                 n_heads=8):
        # d_out = d_model * n_heads
        super().__init__()
        self.d_model = d_model
        self.d_inp = d_inp
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.n_heads = n_heads
        self.d_out = self.d_model * self.n_heads
        
        # cross attention
        self.cross_attn = MHCA(self.n_heads, self.d_inp, self.d_out)
        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.d_inp)
        
        # ffn
        self.linear1 = nn.Linear(self.d_inp, self.d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.d_ffn, self.d_inp)
        self.dropout4 = nn.Dropout(self.dropout)
        self.norm3 = nn.LayerNorm(self.d_inp)
    
    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward(self, query, key, value):
        # cross-attention
        tgt = self.cross_attn(query, key, value)
        query = tgt + self.dropout1(query)
        query = self.norm1(query)
        
        # ffn
        query = self.forward_ffn(query)
        
        return query

def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, \
                                  num_layers)
        self.num_layers = num_layers
        
    def forward(self, query, key, value):
        output = query
        for layer in self.layers:
            output = layer(output, key, value)
        
        return output
    
class MHCA(nn.Module):
    def __init__(self, num_heads, inp_dim, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        # self.v_dim = v_dim
        self.inp_dim = inp_dim
        self.n_heads = num_heads
        
        assert self.hid_dim % self.n_heads == 0
        
        self.w_q = nn.Linear(self.inp_dim, self.hid_dim, bias=False)
        self.w_k = nn.Linear(self.inp_dim, self.hid_dim, bias=False)
        self.w_v = nn.Linear(self.inp_dim, self.hid_dim, bias=False)
        
        self.fc = nn.Linear(self.hid_dim, self.inp_dim)
        self.scale = math.sqrt(self.hid_dim // self.n_heads)
    
    def forward(self, query, key, value, pos_embed=None):
        batch_size, N, C = query.shape # 
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        Q = Q.view(batch_size, N, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, N, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, N, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        
        if pos_embed:
            # pos_embed: bs x N x 2
            self.w_pos = nn.Linear(2, self.hid_dim)
            self.pos_embed = self.w_pos(pos_embed)
            self.pos_embed = self.pos_embed.view(batch_size, N, self.n_heads, self.him_dim // self.n_heads).permute(0, 2, 1, 3)
            Q = Q + self.pos_embed
            K = K + self.pos_embed
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, N, -1)
        x = self.fc(x)
        return x

class MHCA_ein(nn.Module):
    def __init__(self, num_heads, inp_dim, hid_dim, n, pos_embed=True):
        super().__init__()
        self.hid_dim = hid_dim
        # self.v_dim = v_dim
        self.inp_dim = inp_dim
        self.n_heads = num_heads
        self.n = n
        self.pos_embed_bool = pos_embed
        
        assert self.hid_dim % self.n_heads == 0
        
        self.w_q = nn.Linear(self.inp_dim, self.hid_dim, bias=False)
        self.w_k = nn.Linear(self.inp_dim, self.hid_dim, bias=False)
        self.w_v = nn.Linear(self.inp_dim, self.hid_dim, bias=False)
        
        self.fc = nn.Linear(self.hid_dim, self.inp_dim)
        self.scale = math.sqrt(self.hid_dim // self.n_heads)
        self.pos_embed = nn.Parameter(torch.zeros(self.n_heads, self.n, self.n))
    
    def forward(self, query, key, value):
        b, n, m, h = *value.shape, self.n_heads
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        Q = rearrange(Q, "b n (h d) -> b h n d", h=h)
        K = rearrange(K, "b n (h d) -> b h n d", h=h)
        V = rearrange(V, "b n (h d) -> b h n d", h=h)
        
        energy = einsum("b h i d, b h j d -> b h i j", Q, K) / self.scale
        # print("pos_embed_bool", self.pos_embed_bool)
        
        if self.pos_embed is not None and self.pos_embed_bool: 
            energy = energy + self.pos_embed
        attn = torch.softmax(energy, dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, V)
        out = rearrange(out, "b h n d -> b n (h d)", b=b)
        out = self.fc(out)
        return out

def get_topk_pairs(pre_hm, repro_hm, K):
    B_hm, C_hm, H_hm, W_hm = pre_hm.shape # 
    assert pre_hm.shape == repro_hm.shape
    pre_topk = torch.topk(pre_hm.view(B_hm,C_hm,-1), K, dim=-1)[1].float()
    pre_topk_float = pre_topk /  (H_hm * W_hm) 
    repro_topk = torch.topk(repro_hm.view(B_hm, C_hm, -1), K, dim=-1)[1].float()
    repro_topk_float = repro_topk / (H_hm * W_hm)
    return pre_topk_float.view(B_hm, -1), repro_topk_float.view(B_hm, -1), pre_topk.view(B_hm, -1), repro_topk.view(B_hm, -1) # B_hm x K

def get_topk_index(pre_hm, repro_hm, K):
    B_hm, C_hm, H_hm, W_hm = pre_hm.shape 
    assert pre_hm.shape == repro_hm.shape
    pre_topk = torch.topk(pre_hm.flatten(2), K, dim=-1)[1].type(torch.long) # B_hm x 7 x K
    pre_topk = pre_topk.view(B_hm, -1) # B_hm x (C_hm * K)
    pre_topk_indices = torch.zeros(B_hm, C_hm * K, 2)
    pre_topk_indices[:, :, 0] = pre_topk % W_hm
    pre_topk_indices[:, :, 1] = pre_topk // W_hm # B x (C_hm * K) x 2
    
    repro_topk = torch.topk(repro_hm.flatten(2), K, dim=-1)[1].type(torch.long)
    repro_topk = repro_topk.view(B_hm, -1)
    repro_topk_indices = torch.zeros(B_hm, C_hm * K, 2)
    repro_topk_indices[:, :, 0] = repro_topk % W_hm
    repro_topk_indices[:, :, 1] = repro_topk // W_hm
    
    return pre_topk_indices, repro_topk_indices # B_hm x (C_hm * K) x 2
    
def get_topk_features_scale(feats, topk_inds,  scale_num, kernel = 3):
    """

    Parameters
    ----------
    feats : previous features of shape [B, C, H, W]
    topk_inds : pre_topk_indices of shape [B, K, 2]
    kernel : The size of windw, The default is 3.
    scale_num : the scale num of differenct scales of features
    N : kernel ** 2
    Returns
    -------
    neighbor_feats [B, K, N, 2]. Note N = kernel ** 2
    """
    B, C, H, W = feats.shape
    _, K, _ = topk_inds.shape
    assert H == W
    neighbor_coords = torch.arange(-(kernel // 2), (kernel // 2) + 1) 
    k_size = neighbor_coords.shape[0]
    neighbor_coords = torch.flatten(
        torch.stack(torch.meshgrid([neighbor_coords, neighbor_coords]), dim=0),
        1,
    ) # [2, N]  
    # neighbor_coords default are [[-1, -1, -1,  0,  0,  0,  1,  1,  1],
    #        [-1,  0,  1, -1,  0,  1, -1,  0,  1]]
    neighbor_coords = (
        neighbor_coords.permute(1, 0).contiguous().to(topk_inds)
    ) # relative coordinate [N, 2]
    neighbor_coords = (
        topk_inds[:, :, None, :] * scale_num
        + neighbor_coords[None, None, :, :]
    ) # # coordinates [B, K, N, 2]
    
    neighbor_coords = torch.clamp(
        neighbor_coords, min=0, max=H - 1
    ) # prevent out of bound
    
    feat_id = (
                neighbor_coords[:, :, :, 1] * W
                + neighbor_coords[:, :, :, 0]
            )  # pixel id [B, K, N]
    # print('feat_id.shape', feat_id.shape)
    feat_id = feat_id.reshape(B, -1).type(torch.long)  # pixel id [B, K*N]

    batch_id = torch.from_numpy(np.indices((B, K))[0]).type(torch.long) # [B, K]
    
    selected_feat = (
        feats
        .reshape(B, C, -1)
        .permute(0, 2, 1)
        .contiguous()[batch_id.repeat(1, k_size**2), feat_id]
    )  # [B,K * N, C]
    
    return selected_feat, batch_id.repeat(1, k_size**2), feat_id
    

def get_topk_features(pre_features, cur_features, pre_topk, repro_topk):
    B, C, H, W = pre_features.shape
    assert pre_features.shape == cur_features.shape
    pre_topk_int = (pre_topk * H * W).type(torch.long) # size = B x K
    repro_topk_int = (repro_topk * H * W).type(torch.long)
    assert pre_topk.shape == repro_topk.shape
    _, length = pre_topk.shape
    topk_ind1 = torch.arange(B).view(B, 1).repeat(1, length) # size = B x K
    
    pre_f = pre_features.permute(0, 2, 3, 1).contiguous()
    B_q, _, _, C_q = pre_f.shape
    pre_key = pre_f.view(B_q, -1, C_q)[[topk_ind1, pre_topk_int]] # size = B x K x C
    cur_f = cur_features.permute(0, 2, 3, 1).contiguous()
    cur_query = cur_f.view(B_q, -1, C_q)[[topk_ind1, repro_topk_int]]
    
    return pre_key, cur_query

def substitute_topk_features(out, cur_features, repro_topk, mlp):
    # out.shape : B x K x C
    B, C, H, W = cur_features.shape
    repro_topk_int = (repro_topk * H * W).type(torch.long)
    _, length = repro_topk_int.shape
    topk_ind1 = torch.arange(B).view(B, 1).repeat(1, length)
    
    cur_f = cur_features.permute(0, 2, 3, 1).contiguous()
    B_q, H_q, W_q, C_q = cur_f.shape
    cur_f = cur_f.view(B_q, -1, C_q)
    cur_query = cur_f[[topk_ind1, repro_topk_int]] # B x K x C
    cur_out = torch.cat([out, cur_query], dim=-1)
    cur_f[[topk_ind1, repro_topk_int]] = mlp(cur_out)
    cur_f = cur_f.view(B_q, H_q, W_q, C_q)
    cur_f = cur_f.permute(0, 3, 1, 2).contiguous()
    
    return cur_f

def substitute_topk_features_scale(out, cur_features,batch_id, feat_id, mlp):
    # out.shape : B x C x H x W
    # B, C, H, W = cur_features.shape
    cur_f = cur_features.permute(0, 2, 3, 1).contiguous()
    B_q, H_q, W_q, C_q = cur_f.shape
    cur_f = cur_f.view(B_q, -1, C_q)
    cur_query = cur_f[batch_id, feat_id]
    cur_out = torch.cat([out, cur_query], dim=-1)
    cur_f[batch_id, feat_id] = mlp(cur_out)
    cur_f = cur_f.view(B_q, H_q, W_q, C_q)
    cur_f = cur_f.permute(0, 3, 1, 2).contiguous()
    
    return cur_f
    

class DLA_PlanA(BaseModelPlanA):
    def __init__(self, num_layers, heads, head_convs, opt, K=28):
        super(DLA_PlanA, self).__init__(
            heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        down_ratio=4
        print("################## Plan A Start! ######################")
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]
        # print('Using node type:', self.node_type)
        # print('this one')
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        print('num_layers', num_layers)
        self.base = globals()['dla{}'.format(num_layers)](
            pretrained=(opt.load_model == ''), opt=opt)
        
        # print(self.base)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        
#        channels_up = [i * 2 for i in channels]
#        self.dla_up = DLAUp(
#            self.first_level, channels_up[self.first_level:], scales,
#            node_type=self.node_type)
#        out_channel = channels_up[self.first_level]
#
#        self.ida_up = IDAUp(
#            out_channel, channels_up[self.first_level:self.last_level], 
#            [2 ** i for i in range(self.last_level - self.first_level)],
#            node_type=self.node_type)
        
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:], scales,
            node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level], 
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type)
        
        self.transformer = nn.ModuleList(
            [TransformerEncoder(
                TransformerEncoderLayerOri(d_inp=16*(2**i), d_model=4*(2**i)), num_layers=3
                ) for i in range(6)])
        self.cat_layer = nn.ModuleList(
            [nn.Sequential(nn.Linear(16*(2**(i+1)), 32*(2**(i+1))),
                           nn.ReLU(),
                           nn.Linear(32*(2**(i+1)), 16*(2**(i)))) for i in range(6)]
                                      )
        self.K = K

    def imgpre2feats(self, x, pre_img=None, pre_hm=None, repro_hm=None, pre_hm_cls=None, repro_hm_cls=None):
        x_pre = self.base(pre_img=pre_img, pre_hm=pre_hm) 
        x_cur = self.base(pre_img=x, pre_hm = repro_hm) 
        
        # print("################## Plan A Start! ######################")
        x_out = []
        assert len(x_pre) == len(x_cur)
        pre_topk_indices, repro_topk_indices, pre_topk_int, repro_topk_int = get_topk_pairs(pre_hm, repro_hm, self.K)
        for i in range(len(x_cur)):
            pre_features, cur_features = x_pre[i], x_cur[i]
            
            # PlanA plus Transformer 
            transformer = self.transformer[i]
            pre_key, cur_query = get_topk_features(pre_features, cur_features, pre_topk_indices, repro_topk_indices)
            out = transformer(cur_query, pre_key, pre_key)
            out = substitute_topk_features(out, cur_features, repro_topk_indices, self.cat_layer[i])
            # print(out.shape)
            assert out.shape == x_pre[i].shape
            x_out.append(out)


#            out = torch.cat([pre_features, cur_features], dim=1)
#            print(out.shape)
#            x_out.append(out)
          
        x_out = self.dla_up(x_out)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x_out[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]],pre_topk_int, repro_topk_int
        
class DLA_PlanACAT(BaseModelPlanA):
    def __init__(self, num_layers, heads, head_convs, opt, K=28):
        super(DLA_PlanACAT, self).__init__(
            heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        down_ratio=4
        print("################## Plan A CAT Start! ######################")
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]
        # print('Using node type:', self.node_type)
        # print('this one')
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        print('num_layers', num_layers)
        self.base = globals()['dla{}'.format(num_layers)](
            pretrained=(opt.load_model == ''), opt=opt)
        
        # print(self.base)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        channels_up = [i * 2 for i in channels]
        self.dla_up = DLAUp(
            self.first_level, channels_up[self.first_level:], scales,
            node_type=self.node_type)
        out_channel = channels_up[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels_up[self.first_level:self.last_level], 
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type)

    def imgpre2feats(self, x, pre_img=None, pre_hm=None, repro_hm=None, pre_hm_cls = None, repro_hm_cls=None):
        x_pre = self.base(pre_img=pre_img, pre_hm=pre_hm) 
        x_cur = self.base(pre_img=x, pre_hm = repro_hm)
        
        # print("################## Plan A CAT Start! ######################")
        x_out = []
        assert len(x_pre) == len(x_cur)
        # pre_topk_indices, repro_topk_indices, pre_topk_int, repro_topk_int = get_topk_pairs(pre_hm, repro_hm, self.K)
        for i in range(len(x_cur)):
            pre_features, cur_features = x_pre[i], x_cur[i]
            out = torch.cat([pre_features, cur_features], dim=1)
            # print(out.shape)
            x_out.append(out)
          
        x_out = self.dla_up(x_out)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x_out[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]],None, None

class DLA_PlanAAblation(BaseModelPlanA):
    def __init__(self, num_layers, heads, head_convs, opt):
        super(DLA_PlanAAblation, self).__init__(
            heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        down_ratio=4
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]
        print(f"####################{self.opt.phase}#########################")
        # print('Using node type:', self.node_type)
        # print('this one')
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        print('num_layers', num_layers)
        self.base = globals()['dla{}'.format(num_layers)](
            pretrained=(opt.load_model == ''), opt=opt)
        
        if self.opt.phase == "ablation_wo_shared":
            self.base1 = globals()['dla{}'.format(num_layers)](
                pretrained=(opt.load_model == ''), opt=opt)
        
        # print(self.base)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:], scales,
            node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level], 
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type)
        
        self.cat_layer = nn.ModuleList(
            [nn.Sequential(nn.Linear(16*(2**(i+1)), 32*(2**(i+1))),
                           nn.ReLU(),
                           nn.Linear(32*(2**(i+1)), 16*(2**(i)))) for i in range(6)])


    def imgpre2feats(self, x, pre_img=None, pre_hm=None, repro_hm=None, pre_hm_cls = None, repro_hm_cls=None):
        x_out = []
        if self.opt.phase == "ablation_wo_shared":
            x_pre = self.base(pre_img=pre_img, pre_hm=pre_hm)
            x_cur = self.base1(pre_img=x)
            # print("##################### Ablation_wo_shared################### ")
            assert len(x_pre) == len(x_cur)
            for i in range(len(x_cur)):
                pre_feats, cur_feats = x_pre[i], x_cur[i]
                pre_f = pre_feats.permute(0, 2, 3, 1).contiguous()
                cur_f = cur_feats.permute(0, 2, 3, 1).contiguous()
                out = self.cat_layer[i](torch.cat([pre_f, cur_f], dim=-1))
                out = out.permute(0, 3, 1, 2).contiguous()
                assert out.shape == x_pre[i].shape
                x_out.append(out)
        elif self.opt.phase == "ablation_shared":
            x_pre = self.base(pre_img=pre_img, pre_hm=pre_hm)
            x_cur = self.base(pre_img=x)
            # print("##################### Ablation_shared################### ")
            assert len(x_pre) == len(x_cur)
            for i in range(len(x_cur)):
                pre_feats, cur_feats = x_pre[i], x_cur[i]
                pre_f = pre_feats.permute(0, 2, 3, 1).contiguous()
                cur_f = cur_feats.permute(0, 2, 3, 1).contiguous()
                out = self.cat_layer[i](torch.cat([pre_f, cur_f], dim=-1))
                out = out.permute(0, 3, 1, 2).contiguous()
                assert out.shape == x_pre[i].shape
                x_out.append(out)
        elif self.opt.phase == "ablation_shared_repro":
            x_pre = self.base(pre_img=pre_img, pre_hm=pre_hm)
            x_cur = self.base(pre_img=x, pre_hm=repro_hm)
            # print("##################### Ablation_shared_repro################### ")
            assert len(x_pre) == len(x_cur)
            for i in range(len(x_cur)):
                pre_feats, cur_feats = x_pre[i], x_cur[i]
                pre_f = pre_feats.permute(0, 2, 3, 1).contiguous()
                cur_f = cur_feats.permute(0, 2, 3, 1).contiguous()
                out = self.cat_layer[i](torch.cat([pre_f, cur_f], dim=-1))
                out = out.permute(0, 3, 1, 2).contiguous()
                assert out.shape == x_pre[i].shape
                x_out.append(out)
        else:
            raise ValueError
        
        x_out = self.dla_up(x_out)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x_out[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]], None, None



class DLA_PlanAWindow(BaseModelPlanA):
    def __init__(self, num_layers, heads, head_convs, opt):
        super(DLA_PlanAWindow, self).__init__(
            heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        down_ratio=4
        print("################## Plan A Window Start! ######################")
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]
        # print('Using node type:', self.node_type)
        # print('this one')
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        print('num_layers', num_layers)
        print("pretrained", opt.load_model=="")
        self.base = globals()['dla{}'.format(num_layers)](
            pretrained=(opt.load_model == ''), opt=opt)
        
        # print(self.base)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:], scales,
            node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level], 
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type)
        
        self.K_list = [int(opt.k_list_1), int(opt.k_list_2), int(opt.k_list_3)]
        self.kernel_list = [int(opt.ks1), int(opt.ks2), int(opt.ks3)]
        # self.kernel_list = [12, 6, 3]
        # self.kernel_list = [3,3,3] 
        self.scale_list = [4, 2, 1]
        
        self.transformer = nn.ModuleList(
            [TransformerEncoder(
                TransformerEncoderLayer(d_inp=16*(2**i), d_model=4*(2**i), n_k=opt.num_classes*self.K_list[i]*(1 + 2 * (self.kernel_list[i]//2))**2, pos_embed=self.opt.pos_embed), num_layers=3
                 ) for i in range(3)])
        self.cat_layer = nn.ModuleList(
            [nn.Sequential(nn.Linear(16*(2**(i+1)), 32*(2**(i+1))),
                           nn.ReLU(),
                           nn.Linear(32*(2**(i+1)), 16*(2**(i)))) for i in range(6)])
                                      

    def imgpre2feats(self, x, pre_img=None, pre_hm=None, repro_hm=None, pre_hm_cls = None, repro_hm_cls=None):
        x_pre = self.base(pre_img=pre_img, pre_hm=pre_hm) 
        x_cur = self.base(pre_img=x, pre_hm = repro_hm) 
#        x_out = self.base(x, pre_img, pre_hm)
        
        # print("################## Plan A Window Start! ######################")
        x_out = []
        assert len(x_pre) == len(x_cur)
        for i in range(len(x_cur)):
            pre_feats, cur_feats = x_pre[i], x_cur[i]
            if i == 0:
                pre_topk_indices, repro_topk_indices = get_topk_index(pre_hm_cls, repro_hm_cls, self.K_list[i])  # B_hm x (C_hm * K) x 2
                transformer = self.transformer[i]
                pre_key, prev_batch_id, prev_feat_id_1 = get_topk_features_scale(pre_feats, pre_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                cur_query, cur_batch_id, cur_feat_id_1 = get_topk_features_scale(cur_feats, repro_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                out = transformer(cur_query, pre_key, pre_key)
                out = substitute_topk_features_scale(out, cur_feats, cur_batch_id, cur_feat_id_1, self.cat_layer[i])
                # print(out.shape)
                assert out.shape == x_pre[i].shape
                x_out.append(out)
                # x_out.append(pre_feats+cur_feats)
            elif 1 <= i <= 2:
                pre_topk_indices, repro_topk_indices = get_topk_index(pre_hm_cls, repro_hm_cls, self.K_list[i])  # B_hm x (C_hm * K) x 2
                transformer = self.transformer[i]
                pre_key, prev_batch_id, prev_feat_id = get_topk_features_scale(pre_feats, pre_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                cur_query, cur_batch_id, cur_feat_id = get_topk_features_scale(cur_feats, repro_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                out = transformer(cur_query, pre_key, pre_key)
                out = substitute_topk_features_scale(out, cur_feats, cur_batch_id, cur_feat_id, self.cat_layer[i])
                # print(out.shape)
                assert out.shape == x_pre[i].shape
                x_out.append(out)
            else:
                pre_f = pre_feats.permute(0, 2, 3, 1).contiguous()
                cur_f = cur_feats.permute(0, 2, 3, 1).contiguous()
                out = self.cat_layer[i](torch.cat([pre_f, cur_f], dim=-1))
                out = out.permute(0, 3, 1, 2).contiguous()
                assert out.shape == x_pre[i].shape
                x_out.append(out)
                # x_out.append(pre_feats+cur_feats)
           #  x_out.append(pre_feats+cur_feats)
          
        x_out = self.dla_up(x_out)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x_out[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]], prev_feat_id_1, cur_feat_id_1
        
        
class DLA_PlanAWindow_Three(BaseModelPlanA_Three):
    def __init__(self, num_layers, heads, head_convs, opt):
        super(DLA_PlanAWindow_Three, self).__init__(
            heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        down_ratio=4
        print("################## Plan A Window Three Start! ######################")
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]
        # print('Using node type:', self.node_type)
        # print('this one')
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        print('num_layers', num_layers)
        print("pretrained", opt.load_model=="")
        self.base = globals()['dla{}'.format(num_layers)](
            pretrained=(opt.load_model == ''), opt=opt)
        
        # print(self.base)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:], scales,
            node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level], 
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type)
        
        self.K_list = [int(opt.k_list_1), int(opt.k_list_2), int(opt.k_list_3)]
        self.kernel_list = [int(opt.ks1), int(opt.ks2), int(opt.ks3)]
        # self.kernel_list = [12, 6, 3]
        # self.kernel_list = [3,3,3] 
        self.scale_list = [4, 2, 1]
        if not self.opt.shared_ca:
            self.pprev_transformer = nn.ModuleList(
                [TransformerEncoder(
                    TransformerEncoderLayer(d_inp=16*(2**i), d_model=4*(2**i), n_k=opt.num_classes*self.K_list[i]*(1 + 2 * (self.kernel_list[i]//2))**2, pos_embed=self.opt.pos_embed), num_layers=3
                     ) for i in range(3)])
        self.prev_transformer = nn.ModuleList(
            [TransformerEncoder(
                TransformerEncoderLayer(d_inp=16*(2**i), d_model=4*(2**i), n_k=opt.num_classes*self.K_list[i]*(1 + 2 * (self.kernel_list[i]//2))**2, pos_embed=self.opt.pos_embed), num_layers=3
                 ) for i in range(3)])
        self.cat_layer = nn.ModuleList(
            [nn.Sequential(nn.Linear(16*(2**(i))*3, 32*(2**(i))*3),
                           nn.ReLU(),
                           nn.Linear(32*(2**(i))*3, 16*(2**(i)))) for i in range(6)])
                                      

    def imgpre2feats(self, x, ppre_img = None, pre_img=None, 
                     ppre_hm = None, pre_hm=None, repro_hm=None, 
                     ppre_hm_cls = None, pre_hm_cls = None, repro_hm_cls=None):
        
        x_ppre = self.base(pre_img=ppre_img, pre_hm=ppre_hm)
        x_pre = self.base(pre_img=pre_img, pre_hm=pre_hm) 
        x_cur = self.base(pre_img=x, pre_hm = repro_hm) 
#        x_out = self.base(x, pre_img, pre_hm)
        
        # print("################## Plan A Window Start! ######################")
        x_out = []
        assert len(x_pre) == len(x_cur)
        assert len(x_ppre) == len(x_cur)
        for i in range(len(x_cur)):
            ppre_feats, pre_feats, cur_feats = x_ppre[i], x_pre[i], x_cur[i]
            if i <= 2:
                ppre_topk_indices, _ = get_topk_index(ppre_hm_cls, repro_hm_cls, self.K_list[i])
                pre_topk_indices, repro_topk_indices = get_topk_index(pre_hm_cls, repro_hm_cls, self.K_list[i])  # B_hm x (C_hm * K) x 2
                if not self.opt.shared_ca:
                    ppre_transformer = self.pprev_transformer[i]
                pre_transformer = self.prev_transformer[i]
                
                # print(self.pprev_transformer)
                
                ppre_key, pprev_batch_id, pprev_feat_id_1 = get_topk_features_scale(ppre_feats, ppre_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                pre_key, prev_batch_id, prev_feat_id_1 = get_topk_features_scale(pre_feats, pre_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                cur_query, cur_batch_id, cur_feat_id_1 = get_topk_features_scale(cur_feats, repro_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                
                if not self.opt.shared_ca:
                    ppre_out = ppre_transformer(cur_query, ppre_key, ppre_key)
                else:
                    ppre_out = pre_transformer(cur_query, ppre_key, ppre_key)
                pre_out = pre_transformer(cur_query, pre_key, pre_key)
                out = torch.cat([ppre_out, pre_out], dim=-1)
                out = substitute_topk_features_scale(out, cur_feats, cur_batch_id, cur_feat_id_1, self.cat_layer[i])
                # print(out.shape)
                assert out.shape == x_pre[i].shape
                x_out.append(out)
                # x_out.append(pre_feats+cur_feats)
            else:
                ppre_f = pre_feats.permute(0, 2, 3, 1).contiguous()
                pre_f = pre_feats.permute(0, 2, 3, 1).contiguous()
                cur_f = cur_feats.permute(0, 2, 3, 1).contiguous()
                out = self.cat_layer[i](torch.cat([ppre_f, pre_f, cur_f], dim=-1))
                out = out.permute(0, 3, 1, 2).contiguous()
                assert out.shape == x_pre[i].shape
                x_out.append(out)
                # x_out.append(pre_feats+cur_feats)
           #  x_out.append(pre_feats+cur_feats)
          
        x_out = self.dla_up(x_out)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x_out[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]], prev_feat_id_1, cur_feat_id_1   

class DLA_PlanAWindow_l3new(BaseModelPlanA): 
    def __init__(self, num_layers, heads, head_convs, opt):
        super(DLA_PlanAWindow_l3new, self).__init__(
            heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        down_ratio=4
        print("################## Plan A Window Start! ######################")
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]
        # print('Using node type:', self.node_type)
        # print('this one')
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        print('num_layers', num_layers)
        print("pretrained", opt.load_model=="")
        self.base = globals()['dla{}'.format(num_layers)](
            pretrained=(opt.load_model == ''), opt=opt)
        
        # print(self.base)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:], scales,
            node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level], 
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type)
        
        self.K_list = [int(opt.k_list_1), int(opt.k_list_2), int(opt.k_list_3), int(opt.k_list_4), int(opt.k_list_5), int(opt.k_list_6)]
        self.kernel_list = [int(opt.ks1), int(opt.ks2), int(opt.ks3), int(opt.ks4), int(opt.ks5), int(opt.ks6)]
        # self.kernel_list = [12, 6, 3]
        # self.kernel_list = [3,3,3] 
        self.scale_list = [4, 2, 1, 1/2, 1/4, 1/8]
        
        self.transformer = nn.ModuleList(
            [TransformerEncoder(
                TransformerEncoderLayer(d_inp=16*(2**i), d_model=4*(2**i), n_k=opt.num_classes*self.K_list[i]*(1 + 2 * (self.kernel_list[i]//2))**2, pos_embed=self.opt.pos_embed), num_layers=3
                 ) for i in range(3)])
        self.cat_layer = nn.ModuleList(
            [nn.Sequential(nn.Linear(16*(2**(i+1)), 32*(2**(i+1))),
                           nn.ReLU(),
                           nn.Linear(32*(2**(i+1)), 16*(2**(i)))) for i in range(6)])
                                      

    def imgpre2feats(self, x, pre_img=None, pre_hm=None, repro_hm=None, pre_hm_cls = None, repro_hm_cls=None):
        x_pre = self.base(pre_img=pre_img, pre_hm=pre_hm) 
        x_cur = self.base(pre_img=x, pre_hm = repro_hm) 
#        x_out = self.base(x, pre_img, pre_hm)
        
        # print("################## Plan A Window Start! ######################")
        x_out = []
        assert len(x_pre) == len(x_cur)
        for i in range(len(x_cur)):
            pre_feats, cur_feats = x_pre[i], x_cur[i]
            if i == 0:
                pre_topk_indices, repro_topk_indices = get_topk_index(pre_hm_cls, repro_hm_cls, self.K_list[i])  # B_hm x (C_hm * K) x 2
                transformer = self.transformer[i]
                pre_key, prev_batch_id, prev_feat_id_1 = get_topk_features_scale(pre_feats, pre_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                cur_query, cur_batch_id, cur_feat_id_1 = get_topk_features_scale(cur_feats, repro_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                out = transformer(cur_query, pre_key, pre_key)
                out = substitute_topk_features_scale(out, cur_feats, cur_batch_id, cur_feat_id_1, self.cat_layer[i])
                # print(out.shape)
                assert out.shape == x_pre[i].shape
                x_out.append(out)
                # x_out.append(pre_feats+cur_feats)
            elif 1 <= i <= 2:
                pre_topk_indices, repro_topk_indices = get_topk_index(pre_hm_cls, repro_hm_cls, self.K_list[i])  # B_hm x (C_hm * K) x 2
                transformer = self.transformer[i]
                pre_key, prev_batch_id, prev_feat_id = get_topk_features_scale(pre_feats, pre_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                cur_query, cur_batch_id, cur_feat_id = get_topk_features_scale(cur_feats, repro_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                out = transformer(cur_query, pre_key, pre_key)
                out = substitute_topk_features_scale(out, cur_feats, cur_batch_id, cur_feat_id, self.cat_layer[i])
                # print(out.shape)
                assert out.shape == x_pre[i].shape
                x_out.append(out)
            else:
                pre_topk_indices, repro_topk_indices = get_topk_index(pre_hm_cls, repro_hm_cls, self.K_list[i])  # B_hm x (C_hm * K) x 2
                pre_key, prev_batch_id, prev_feat_id = get_topk_features_scale(pre_feats, pre_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                cur_query, cur_batch_id, cur_feat_id = get_topk_features_scale(cur_feats, repro_topk_indices, scale_num=self.scale_list[i], kernel=self.kernel_list[i])
                #print("pre_key.shape", pre_key.shape)        v        
                out = substitute_topk_features_scale(pre_key, cur_feats, cur_batch_id, cur_feat_id, self.cat_layer[i])
                assert out.shape == x_pre[i].shape
                x_out.append(out)

                # x_out.append(pre_feats+cur_feats)
           #  x_out.append(pre_feats+cur_feats)
          
        x_out = self.dla_up(x_out)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x_out[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]], prev_feat_id_1, cur_feat_id_1
