# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.checkpoint as cp
from collections import OrderedDict
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from edgeml_pytorch.graph.rnnpool import *
from edgeml_pytorch.graph.rnn import *

__all__ = ['MobileNetV2', 'mobilenetv2_rnnpool']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class ModifiedRNNPool(RNNPool):
    """Use LSTM to modify RNNPool."""

    def __init__(self, nRows, nCols, nHiddenDims,
                 nHiddenDimsBiDir, inputDims, 
                 w1Sparsity=1.0, u1Sparsity=1.0, w2Sparsity=1.0, u2Sparsity=1.0):
        super(ModifiedRNNPool, self).__init__(nRows, nCols, nHiddenDims,
                                              nHiddenDimsBiDir, inputDims,
                                              w1Sparsity, u1Sparsity, w2Sparsity, u2Sparsity)

    def _build(self):
        self.cell_rnn = LSTM(self.inputDims, self.nHiddenDims, gate_nonlinearity="sigmoid",
                             update_nonlinearity="tanh", wSparsity=self.w1Sparsity, uSparsity=self.u1Sparsity,
                             batch_first=False, bidirectional=False)

        self.cell_bidirrnn = LSTM(self.nHiddenDims, self.nHiddenDimsBiDir, gate_nonlinearity="sigmoid",
                                  update_nonlinearity="tanh", wSparsity=self.w2Sparsity, uSparsity=self.u2Sparsity,
                                  batch_first=False, bidirectional=True, is_shared_bidirectional=True)

    def static_single(self,inputs, hidden, batch_size):

        outputs = self.cell_rnn(inputs, hidden[0], hidden[1])
        return torch.split(outputs[0][-1], split_size_or_sections=batch_size, dim=0)

    def forward(self,inputs,batch_size):
        ## across rows

        row_timestack = torch.cat(torch.unbind(inputs, dim=3),dim=0) 

        stateList = self.static_single(torch.stack(torch.unbind(row_timestack,dim=2)),
                        (torch.zeros(1, batch_size * self.nRows, self.nHiddenDims).to(inputs.device),
                        torch.zeros(1, batch_size * self.nRows, self.nHiddenDims).to(inputs.device)),batch_size)       

        outputs_cols = self.cell_bidirrnn(torch.stack(stateList),
                        torch.zeros(2, batch_size, self.nHiddenDimsBiDir).to(inputs.device),
                        torch.zeros(2, batch_size, self.nHiddenDimsBiDir).to(inputs.device))


        ## across columns
        col_timestack = torch.cat(torch.unbind(inputs, dim=2),dim=0)

        stateList = self.static_single(torch.stack(torch.unbind(col_timestack,dim=2)),
                        (torch.zeros(1, batch_size * self.nRows, self.nHiddenDims).to(inputs.device),
                        torch.zeros(1, batch_size * self.nRows, self.nHiddenDims).to(inputs.device)),batch_size)

        outputs_rows = self.cell_bidirrnn(torch.stack(stateList),
                        torch.zeros(2, batch_size, self.nHiddenDimsBiDir).to(inputs.device),
                        torch.zeros(2, batch_size, self.nHiddenDimsBiDir).to(inputs.device))

        output = torch.cat([outputs_rows[0][-1],outputs_cols[0][-1]],1)

        return output


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.01),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.01),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, 
                 num_classes=1000, 
                 width_mult=0.5, 
                 inverted_residual_setting=None, 
                 round_nearest=8,
                 block=None,
                 last_channel = 1280):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()
        
        if block is None:
            block = InvertedResidual
        
        # jk modified
        # input_channel = 8
        input_channel = 32
        #last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # jk modified
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.features_init = ConvBNReLU(3, input_channel, stride=2)

        # jk modified
        # self.unfold = nn.Unfold(kernel_size=(6,6),stride=(4,4))

        # jk modified
        # self.rnn_model = RNNPool(6, 6, 8, 8, input_channel)#num_init_features)
        # self.fold = nn.Fold(kernel_size=(1,1),output_size=(27,27))

        self.rnn_model_end = ModifiedRNNPool(7, 7, int(self.last_channel/4), int(self.last_channel/4), self.last_channel)

        features=[] 

        input_channel = 32

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            #nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.features_init(x)
     
        # jk modified, remove rnnpool used in the front of mobilenetv2
        # patches = self.unfold(x)
        # patches = torch.cat(torch.unbind(patches,dim=2),dim=0)
        # patches = torch.reshape(patches,(-1,8,6,6))
        

        # output_x = int((x.shape[2]-6)/4 + 1)
        # output_y = int((x.shape[3]-6)/4 + 1)

        # rnnX = self.rnn_model(patches, int(batch_size)*output_x*output_y)

        # x = torch.stack(torch.split(rnnX, split_size_or_sections=int(batch_size), dim=0),dim=2)

        # x = self.fold(x)

        # x = F.pad(x, (0,1,0,1), mode='replicate')        

        x = self.features(x)
        x = self.rnn_model_end(x, batch_size)
        x = self.classifier(x)
        return x


def mobilenetv2_rnnpool(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
