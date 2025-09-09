'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706

The code is for ResNet34 backbone.
'''

import os
import logging
from typing import Union
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ResNet34(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet34, self).__init__()
        """ Constructor
        Args:
            resnet_config: configuration file with the dict format
        """
        self.num_classes = num_classes
        self.mode = '11'

        # Define layers of the backbone
        resnet = torchvision.models.resnet34(pretrained=True)  # FIXME: download the pretrained weights from online
        # resnet.conv1 = nn.Conv2d(inc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

        if self.mode == 'adjust_channel':
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(512, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )


    def features(self, inp):
        x = self.resnet(inp)
        return x

    def classifier(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, inp):
        x = self.features(inp)
        out = self.classifier(x)
        return out


def resnet(num_classes: int = 2, weight_path: str = None, pre_imgnet: bool = False, **kwargs):
    model = ResNet34(num_classes)
    if weight_path is not None:
        if pre_imgnet:
            checkpoint = torch.load(weight_path)
            pretrained_dict = torch.load(weight_path)
            if num_classes!=0:
                fc = nn.Linear(2048, num_classes)
            pretrained_dict["fc.weight"] = fc.weight
            pretrained_dict["fc.bias"] = fc.bias
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=True)
        else:
            model.load_state_dict(torch.load(weight_path), strict=False)
    return model