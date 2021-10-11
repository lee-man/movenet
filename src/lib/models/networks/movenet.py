# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# Modified by Min Li
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import cv2

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from .backbone_utils import mobilenet_backbone
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


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

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class MoveNet(nn.Module):
    '''
    MoveNet from Goolge. Please refer their blog: https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html

    '''
    def __init__(self, backbone, heads, head_conv, ft_size=48):
        super(MoveNet, self).__init__()
        self.out_channels = 24
        self.backbone = backbone
        self.heads = heads
        self.ft_size = ft_size
        # self.weight_to_center = self._generate_center_dist(self.ft_size).unsqueeze(2)
 
        # self.dist_y, self.dist_x = self._generate_dist_map(self.ft_size)
        # self.index_17 = torch.arange(0, 17).float()

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, groups=self.out_channels, bias=True),
                  nn.Conv2d(self.out_channels, head_conv, 1, 1, 0, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
                # if 'hm' in head:
                #     fc[-1].bias.data.fill_(-2.19)
                # else:
                #     fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                # if 'hm' in head:
                #     fc.bias.data.fill_(-2.19)
                # else:
                #     fill_fc_weights(fc)
            self.__setattr__(head, fc)


    def forward(self, x):
        # conv forward
        # x  = x * 0.007843137718737125 - 1.0
        x = self.backbone(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)

        return [ret]

    def decode(self, x):

        kpt_heatmap, center, kpt_regress, kpt_offset = x['hm_hp'].permute((1, 2, 0)), x['hm'].permute((1, 2, 0)), x['hps'].permute((1, 2, 0)), x['hp_offset'].permute((1, 2, 0))

        # pose decode
        kpt_heatmap = torch.sigmoid(kpt_heatmap)
        center = torch.sigmoid(center)

        ct_y, ct_x = self._top_with_center(center, self.ft_size)
        ct_y, ct_x = ct_y.squeeze(0).type(torch.LongTensor), ct_x.squeeze(0).type(torch.LongTensor)
        kpt_ys_regress, kpt_xs_regress = self._center_to_kpt(kpt_regress, ct_y, ct_x)
        kpt_ys_heatmap, kpt_xs_heatmap = self._kpt_from_heatmap(kpt_heatmap, kpt_ys_regress, kpt_xs_regress, self.ft_size)

        kpt_with_conf = self._kpt_from_offset(kpt_offset, kpt_ys_heatmap, kpt_xs_heatmap, kpt_heatmap, self.ft_size)
        
        return kpt_with_conf

        
    def _draw(self, ft):
        plt.imshow(ft.numpy().reshape(self.ft_size, self.ft_size))
        # img = (data-np.min(data))/(np.max(data)-np.min(data))*255
        plt.show()

    def _generate_center_dist(self, ft_size=48, delta=1.8):
        weight_to_center = torch.zeros((int(ft_size), int(ft_size)))
        y, x = np.ogrid[0:ft_size, 0:ft_size]
        center_y, center_x = ft_size / 2.0, ft_size/ 2.0
        y = y - center_y
        x = x - center_x
        # weight_to_center = 1 / (np.sqrt(np.abs(x) + np.abs(y)) + delta)
        weight_to_center = 1 / (np.sqrt(y * y + x * x) + delta)
        weight_to_center = torch.from_numpy(weight_to_center)
        return weight_to_center

    def _generate_dist_map(self, ft_size=48):
        y, x = np.ogrid[0:ft_size, 0:ft_size]
        y = torch.from_numpy(np.repeat(y, ft_size, axis=1)).unsqueeze(2).float()
        x = torch.from_numpy(np.repeat(x, ft_size, axis=0)).unsqueeze(2).float()

        return y, x


    def _top_with_center(self, center, size=48):
        scores = center * self.weight_to_center

        top_indx = torch.argmax(scores.view(1, 48 * 48, 1), dim=1)
        # top_y = torch.div(top_indx, size, rounding_mode='floor')
        top_y = (top_indx / size).int().float()
        top_x = top_indx - top_y * size

        return top_y, top_x

    def _center_to_kpt(self, kpt_regress, ct_y, ct_x):
        kpt_coor = kpt_regress[ct_y, ct_x, :] #.squeeze(0)
        # kpt_coor = kpt_coor.reshape((17, 2))
        ys, xs = kpt_coor[0, :17] + ct_y.float(), kpt_coor[0, 17:] + ct_x.float()
        
        return (ys, xs)

    def _kpt_from_heatmap(self, kpt_heatmap, kpt_ys, kpt_xs, size=48):
        y = self.dist_y - kpt_ys.reshape(1, 1, 17)
        x = self.dist_x - kpt_xs.reshape(1, 1, 17)
        dist_weight = torch.sqrt(y * y + x * x) + 1.8
        
        scores = kpt_heatmap / dist_weight
        scores = scores.reshape((1, 48 * 48, 17))
        top_inds = torch.argmax(scores, dim=1)
        # kpts_ys = torch.div(top_inds, size, rounding_mode='floor')
        kpts_ys = (top_inds / size).int().float()
        kpts_xs = top_inds - kpts_ys * size
        return kpts_ys, kpts_xs
    
    def _kpt_from_offset(self, kpt_offset, kpts_ys, kpts_xs, kpt_heatmap, size=48):
        kpt_offset = kpt_offset.reshape(size, size, 17, 2)
        kpt_coordinate = torch.stack([kpts_ys.squeeze(0), kpts_xs.squeeze(0)], dim=1)

        kpt_offset_yx = torch.zeros((17, 2))
        kpt_conf = torch.zeros((17, 1))

        kpt_offset_yx = kpt_offset[kpt_coordinate[:, 0].type(torch.LongTensor), kpt_coordinate[:, 1].type(torch.LongTensor), self.index_17.type(torch.LongTensor), :]
        kpt_conf = kpt_heatmap[kpt_coordinate[:, 0].type(torch.LongTensor), kpt_coordinate[:, 1].type(torch.LongTensor), self.index_17.type(torch.LongTensor)].reshape(17, 1)

        kpt_coordinate= (kpt_offset_yx + kpt_coordinate) * 0.02083333395421505
        kpt_with_conf = torch.cat([kpt_coordinate, kpt_conf], dim=1).reshape((1, 1, 17, 3))

        return kpt_with_conf




def get_pose_net(heads, head_conv=96, froze_backbone=True):
    backbone = mobilenet_backbone('mobilenet_v2', pretrained=False, fpn=True)
    if froze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    model = MoveNet(backbone, heads, head_conv=head_conv)
    return model