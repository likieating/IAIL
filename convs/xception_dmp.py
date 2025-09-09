"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import numpy as np

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975
            # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, distance='l2', num_classes=1000, stage=0):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.criterion1 = nn.NLLLoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.distance = distance
        self.num_classes = num_classes

        # self.conv1 = nn.Conv2d(15,32,3,2,0,bias=False)
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.proj = nn.Linear(2048, 128)

        self.k = 4
        self.stage = stage + 2
        num_prototypes = self.k * 5
        # prototypes = nn.Parameter(torch.zeros(self.k*5, 128))
        orth_vec = generate_random_orthogonal_matrix(128, num_prototypes)
        i_nc_nc = torch.eye(num_prototypes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_prototypes, num_prototypes), (1 / num_prototypes))
        prototypes = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                               math.sqrt(num_prototypes / (num_prototypes - 1)))

        self.register_buffer('prototypes', prototypes)
        print(self.prototypes.shape)

    def set_stage(self, cur_stage):
        self.stage = cur_stage + 2
        # self.stage = 2
        # fixed for ablation study

    def features_test(self, input):
        x = self.conv1(input)  # (32, 299, 299)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)  # (64, 299, 299)
        return x

    def features(self, input):
        x = self.conv1(input)  # (32, 299, 299)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)  # (64, 299, 299)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)  # (1024, 299, 299)

        x = self.conv3(x)  # (1536, 299, 299)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)  # (2048, 299, 299)
        x = self.bn4(x)
        return x

    def logits(self, feat, label, return_feat=False):
        feat = self.relu(feat)

        feat = F.adaptive_avg_pool2d(feat, (1, 1))
        feat = feat.view(feat.size(0), -1)
        y_pred = self.last_linear(feat)
        feat = self.proj(feat)

        # bs*c*c*k = bs*k
        if self.distance == 'l2':
            prototypes = F.normalize(self.prototypes, dim=1)
            similarities = -euclidean_dist(feat, prototypes.t())
            similarities = similarities[:, :self.k * self.stage]

            class1_similarities = similarities[:, :self.k].mean(dim=1)
            class2_similarities = similarities[:, self.k:self.k * self.stage].mean(dim=1)

            x = torch.stack([class1_similarities, class2_similarities], dim=1)

            # x = F.softmax(class_similarities, dim=1)
        elif self.distance == 'cos':
            feat = F.normalize(feat, dim=1)
            prototypes = F.normalize(self.prototypes, dim=1)
            # print(feat.shape, prototypes.shape)
            similarities = feat @ prototypes
            similarities = similarities[:, :self.k * self.stage]

            class1_similarities = similarities[:, :self.k].mean(dim=1)
            class2_similarities = similarities[:, self.k:self.k * self.stage].mean(dim=1)
            p_distance = F.softmax(similarities, dim=1)
            x = torch.stack([class1_similarities, class2_similarities], dim=1)

            # x = F.softmax(class_similarities, dim=1)
        else:
            print('invalid self.distance')
        feat = feat.unsqueeze(dim=1)
        # feat = F.normalize(feat, dim=2)
        if label is None:
            return x, feat
        if not return_feat:
            x = torch.log_softmax(x, dim=1)
            ploss = self.criterion1(x, label)
            closs = self.criterion2(y_pred, label)
            return ploss, closs, p_distance, x
        else:
            x_ = torch.log_softmax(x, dim=1)
            ploss = self.criterion1(x_, label)
            closs = self.criterion2(y_pred, label)
            return ploss, closs, p_distance, x, feat

    def forward(self, input, label=None, return_feat=False):
        x = self.features(input)
        return self.logits(x, label=label, return_feat=return_feat)


def xception(num_classes=2, pretrained='imagenet', stage=0, distance='cos'):
    print('using xception_proto with distance:', distance)
    model = Xception(distance=distance, num_classes=num_classes, stage=stage)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        # assert num_classes == settings['num_classes'], \
        #     "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        # model.load_state_dict(model_zoo.load_url(settings['url']))
        state_dict = torch.load(
            '/home/tangshuai/dmp-bk/pre_weights/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        model.load_state_dict(state_dict, strict=False)

    model.last_linear = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(2048, num_classes)
    )
    # del model.fc
    return model