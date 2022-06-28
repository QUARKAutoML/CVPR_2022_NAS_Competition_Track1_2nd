from __future__ import division
from __future__ import print_function

import os
import hashlib
import paddle
import paddle.nn as nn

import random
random.seed(123)


def make_divisible(v, divisor, min_val=None):
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class StemBlock(nn.Layer):
    def __init__(self, inplanes=3, outplanes=64, kernel_size=7, stride=2, padding=3, bias_attr=False):
        super().__init__()
        self.conv = nn.Conv2D(inplanes, outplanes, kernel_size=kernel_size, stride=stride, 
                               padding=padding, bias_attr=bias_attr)
        self.bn = nn.BatchNorm2D(outplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class HeadBlock(nn.Layer):
    def __init__(self, inplanes, num_classes=1000):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(inplanes, num_classes)
    
    def forward(self, x):
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        return x
    

class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, planes=[], stride=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        self.conv1 = nn.Conv2D(planes[0], planes[1], 3, padding=1, stride=stride, bias_attr=False)
        self.bn1 = norm_layer(planes[1])
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes[1], planes[2], 3, padding=1, bias_attr=False)
        self.bn2 = norm_layer(planes[2])
        self.stride = stride

        if stride != 1 or planes[0] != planes[2]:
            self.downsample = nn.Sequential(
                nn.Conv2D(planes[0], planes[2], 1, stride=stride, bias_attr=False),
                norm_layer(planes[2]))
        else:
            self.downsample = None


    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x

class BottleneckBlock(nn.Layer):

    expansion = 4

    def __init__(self, planes=[], stride=1, norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        self.conv1 = nn.Conv2D(planes[0], planes[1], 1, bias_attr=False)
        self.bn1 = norm_layer(planes[1])

        self.conv2 = nn.Conv2D(planes[1], planes[2], 3, padding=1, stride=stride, groups=1, dilation=1, bias_attr=False)
        self.bn2 = norm_layer(planes[2])

        self.conv3 = nn.Conv2D(planes[2], planes[3], 1, bias_attr=False)
        self.bn3 = norm_layer(planes[3])
        self.relu = nn.ReLU()
        self.stride = stride

        if stride != 1 or planes[0] != planes[3]:
            self.downsample = nn.Sequential(
                nn.Conv2D(planes[0], planes[3], 1, stride=stride, bias_attr=False),
                norm_layer(planes[3]))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class Model(nn.Layer):
    def __init__(self, 
        arch, 
        config={"i": [224], "d": [[2, 5], [2, 5], [2, 8], [2, 5]], "k": [3], "c": [1.0, 0.95, 0.90, 0.85, 0.8, 0.75, 0.7]}, 
        block='basic', base_channels=[64, 128, 256, 512], num_classes=1000
        ):
        super(Model, self).__init__()

        self.im_size_dict = {i: x for i, x in enumerate(config['i'], 1)}
        self.depth_dict = {k: k for s, e in config['d'] for k in range(s, e+1)}
        self.kernel_dict = {i: x for i, x in enumerate(config['k'], 1)}
        self.channel_dict = {i: x for i, x in enumerate(config['c'], 1)}

        self.arch = arch

        if block == 'basic':
            block_conv_num = 2
            block = BasicBlock
        elif block == 'bottle':
            block_conv_num = 3
            block = BottleneckBlock
        else:
            raise NotImplementedError
        
        im_size_code = arch[0]
        depth_code = arch[1:5]
        conv0_code = arch[5]
        blocks_code = arch[6:]
        self.im_size = self.im_size_dict[int(im_size_code)]
        self.depth_list = [int(x) for x in depth_code]
        conv0_channel = make_divisible(base_channels[0] * self.channel_dict[int(conv0_code)], 8)
        
        self.num_classes = num_classes
        self._norm_layer = nn.BatchNorm2D

        stride_list = [1, 2, 2, 2]
        self.blocks = nn.LayerList([StemBlock(3, conv0_channel, 7, 2, 3, False)])
        
        in_channel = conv0_channel
        for d, base_ch, s in zip(self.depth_list, base_channels, stride_list):
            idx = 0
            for c in blocks_code:
                if c == '0':
                    idx += 1
                else:
                    break
            blocks_code = blocks_code[idx:]
            code_str = blocks_code[:d*block_conv_num]
            blocks_code = blocks_code[d*block_conv_num:]
            for _ in range(d):
                codes = code_str[:block_conv_num]
                code_str = code_str[block_conv_num:]
                planes = [in_channel]
                for c, exp in zip(codes, [1, 1, block.expansion]):
                    planes.append(make_divisible(base_ch * exp * self.channel_dict[int(c)], 8))
                self.blocks.append(block(planes, s, self._norm_layer))
                in_channel = planes[-1]
                s = 1
        self.blocks.append(HeadBlock(in_channel, num_classes=1000))

        self.super_modules=[]

    def get_archs_samplers(self, archs, archs_num_needed, sample_num_per_step):
        total_steps = archs_num_needed
        total_archs = []
        cur = 0
        while cur<total_steps:
            cur += len(archs)
            random.shuffle(archs)
            total_archs += archs

        print('################ before clip',len(total_archs))
        self.samplers_total_archs = []
        sample_num_per_step_actually = sample_num_per_step
        for i in range(sample_num_per_step_actually):
            self.samplers_total_archs.append(total_archs[(total_steps//sample_num_per_step_actually)*i:(total_steps//sample_num_per_step_actually)*(i+1)])

        for i in range(sample_num_per_step_actually):
            print('################ after clip sampler{}_total_archs:{}'.format(i, len(self.samplers_total_archs[i])))

    # def forward(self, x):
    #     for b in self.blocks:
    #         x = b(x)
    #     return x

    def clean_forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


    def forward(self, x, arch):
        # set cur_blocks
        cur_blocks=[self.blocks[0]]

        depth_code = arch[:4]
        # conv0_code = arch[4]
        blocks_code = arch[4:]

        depth_end=[5, 5, 8, 5]
        cur=1
        for i in range(len(depth_end)):
            for j in range(1, int(depth_end[i])+1):
                if j<=int(depth_code[i]):
                    cur_blocks.append(self.blocks[cur])
                cur+=1
        cur_blocks.append(self.blocks[cur])

        # broadcast channel
        for k in [38, 22, 12]:
            blocks_code = blocks_code[:k+1]+blocks_code[k]+blocks_code[k+1:]

        # set channels
        assert len(blocks_code)==len(self.super_modules)
        for i in range(len(blocks_code)):
            if i<11:
                base_ch=64
            elif i<22:
                base_ch=128
            elif i<39:
                base_ch=256
            elif i<50:
                base_ch=512

            if blocks_code[i]=='0':
                self.super_modules[i].cur_out_channels=None
            else:
                self.super_modules[i].cur_out_channels=make_divisible(base_ch * self.channel_dict[int(blocks_code[i])], 8)

        for b in cur_blocks:
                x = b(x)

        return x

class Few_Shot_Depth_Model(nn.Layer):
    def __init__(self, arch, num=4):
        super(Few_Shot_Depth_Model, self).__init__()

        self.stage_depth=num
        self.split_models = nn.LayerList()
        for _ in range(self.stage_depth):
            self.split_models.append(Model(arch))
        self.split_stage=None

    def get_archs_samplers(self, archs, archs_num_needed, sample_num_per_step):
        total_steps = archs_num_needed
        total_archs = []
        cur = 0
        while cur<total_steps:
            cur += len(archs)
            random.shuffle(archs)
            total_archs += archs

        print('################ before clip',len(total_archs))
        self.samplers_total_archs = []
        sample_num_per_step_actually = sample_num_per_step
        for i in range(sample_num_per_step_actually):
            self.samplers_total_archs.append(total_archs[(total_steps//sample_num_per_step_actually)*i:(total_steps//sample_num_per_step_actually)*(i+1)])

        for i in range(sample_num_per_step_actually):
            print('################ after clip sampler{}_total_archs:{}'.format(i, len(self.samplers_total_archs[i])))

    def load_state_dict(self, state_dict):
        new_state_dict={}

        for i in range(self.stage_depth):
            for key in state_dict:
                new_state_dict['split_models.{}.'.format(i)+key]=state_dict[key]

        self.set_state_dict(new_state_dict)

    def get_supernet_idx(self, arch):
        if self.split_stage==34:
            mapping = { '22': 0,  '23': 0, '24': 0, \
                        '32': 1,  '42': 1, '52': 1, '62':1, '72': 1, '82':1, \
                        '83': 2,  '84': 2, '85': 2, \
                        '25': 3,  '35': 3, '45': 3, '55':3, '65': 3, '75':3, \
                        }
            stage_depth_code=arch[2:4]
            if stage_depth_code in mapping:
                idx = mapping[stage_depth_code]
            else:
                # idx = -1
                idx = 4
        elif self.split_stage==4:
            stage_depth_code = arch[self.split_stage-1]
            depth = int(stage_depth_code)

            idx = depth-2
        else:
            assert False

        return idx

    def forward(self, x, arch):
        idx=self.get_supernet_idx(arch)

        return self.split_models[idx](x, arch)