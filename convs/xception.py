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
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        # self.conv1 = nn.Conv2d(15,32,3,2,0,bias=False)
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # self.adapter=Adapter(728,300)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

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

        self.fc = nn.Linear(2048, 2)
        self.fc_pro = nn.Linear(2048, 128)
        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

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
        # tmp=self.adapter(x)
        # x=x+tmp
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        # tmp = self.adapter(x)
        # x = x + tmp
        x = self.block8(x)
        x = self.block9(x)
        # tmp = self.adapter(x)
        # x = x + tmp
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)  # (1024, 299, 299)

        x = self.conv3(x)  # (1536, 299, 299)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)  # (2048, 299, 299)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        features = x.view(x.size(0), -1)
        return features

    @property
    def feature_dim(self):
        return self.fc.in_features

    def logits(self, features):
        x = self.fc(features)
        return x

    def logits_fc_pro(self,features):
        x=self.fc_pro(features)
        return x

    def forward(self, input):
        features = self.features(input)
        logits = self.logits(features)
        # logits_pro=self.logits_fc_pro(features)
        # return {'logits': logits,'features': features }
        # return logits
        return logits,features

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    def freeze_fc(self):
        for name, param in self.named_parameters():
            if 'fc_pro' not in name:
                param.requires_grad = False




def xception(num_classes: int = 2, weight_path: str = None, pre_imgnet: bool = False, **kwargs):
    model = Xception(num_classes)
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


#
# class WindowAttention(nn.Module):
#     """
#     Args:
#         dim:Number of input channels
#         window_size: the height and width of the window
#         num_heads:Number of attention haeds
#
#     """
#     def __init__(self,dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.dim=dim
#         self.num_heads=num_heads
#         self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
#         self.qkv=nn.Linear(dim,dim*3)
#         if qkv_bias:
#             self.q_bias = nn.Parameter(torch.zeros(dim))
#             self.v_bias = nn.Parameter(torch.zeros(dim))
#         else:
#             self.q_bias = None
#             self.v_bias = None
#         self.attn_drop=nn.Dropout(attn_drop)
#         self.proj=nn.Linear(dim,dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self,x):
#         """
#         Args:
#             x:(num_windows*B,N,C)
#         """
#         B_, N, C = x.shape
#         qkv_bias = None
#         if self.q_bias is not None:
#             qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
#         qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
#         qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         # cosine attention
#         attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
#
#         logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01,device=self.logit_scale.device))).exp()
#         attn = attn * logit_scale
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# class channel_down(nn.Module):
#     def __init__(self,in_channels,out_channels,
#                  use_prune_channel=False,prune_fraction=0.8):
#         super().__init__()
#         if use_prune_channel:
#             conv_layer = getattr(model, layer_name)
#             weight=conv_layer.weight.data
#             l1_norm = weight.abs().sum(dim=(1, 2, 3))  # 对每个通道计算L1范数
#             # 选择最小L1范数的通道进行剪枝
#             num_channels_to_prune = int(prune_fraction * weight.size(0))
#             _, prune_indices = torch.topk(l1_norm, num_channels_to_prune, largest=False)
#
#             # 设置剪枝的通道的权重为零
#             with torch.no_grad():
#                 conv_layer.weight.data[prune_indices] = 0.0
#                 if conv_layer.bias is not None:
#                     conv_layer.bias.data[prune_indices] = 0.0  # 可选，清零bias
#         else:
#             self.conv1x1=nn.Conv2d(in_channels,out_channels,kernel_size=1)
#     def forward(self,x):
#         x=self.conv1x1(x)
#         # B Ph*Pw C
#         x=x.flatten(2).transpose(1,2)
#         return x
#
#
#
# class Adapter(nn.Module):
#     def __init__(self,in_channels,out_channels,
#                  use_prune_channel=False,input_size=14,num_heads=3,
#                  s=0.1):
#         super().__init__()
#
#         self.channel_down=channel_down(in_channels,out_channels)
#         self.channel_up= nn.Conv2d(out_channels, in_channels, kernel_size=1)
#         self.dim=out_channels
#         self.trans=TransformerBlock(self.dim,input_size,num_heads)
#         self.layers=nn.ModuleList()
#         self.layers.append(
#             self.channel_down
#         )
#         self.layers.append(
#             self.trans)
#
#         self.s=s
#
#
#
#     def forward(self,x):
#
#         for layer in self.layers:
#             x=layer(x)
#         x=self.channel_up(x)
#         x=x*self.s
#
#         return x
#
#
#
# class TransformerBlock(nn.Module):
#     def __init__(self,dim,input_size,num_heads,qkv_bias=True,
#                  drop_path=0.,norm_layer=nn.LayerNorm,):
#         super().__init__()
#         self.input_resolution=(input_size,input_size)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.attn=WindowAttention(dim,num_heads)
#         self.norm1=norm_layer(dim)
#
#     def forward(self,x):
#         H,W=self.input_resolution
#         B,L,C=x.shape
#         shortcut = x
#         attn_x=self.attn(x)
#         x=self.drop_path(self.norm1(x))
#         x=x.view(B,H,W,C).permute(0,3,1,2)
#         return x
