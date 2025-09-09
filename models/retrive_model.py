
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from convs.xception_rm import xception

class Texture_extract(nn.Module):
    def __init__(self,num_features,num_attentions):
        super().__init__()
        self.output_features=num_features
        self.M = num_attentions
        self.conv_extract = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv0 = nn.Conv2d(num_features * num_attentions, num_features * num_attentions, 5, padding=2,
                               groups=num_attentions)
        self.conv1 = nn.Conv2d(num_features * num_attentions, num_features * num_attentions, 3, padding=1,
                               groups=num_attentions)
        self.bn1 = nn.BatchNorm2d(num_features * num_attentions)
        self.conv2 = nn.Conv2d(num_features * 2 * num_attentions, num_features * num_attentions, 3, padding=1,
                               groups=num_attentions)
        self.bn2 = nn.BatchNorm2d(2 * num_features * num_attentions)
        self.conv3 = nn.Conv2d(num_features * 3 * num_attentions, num_features * num_attentions, 3, padding=1,
                               groups=num_attentions)
        self.bn3 = nn.BatchNorm2d(3 * num_features * num_attentions)
        self.conv_last = nn.Conv2d(num_features * 4 * num_attentions, num_features * num_attentions, 1,
                                   groups=num_attentions)
        self.bn4 = nn.BatchNorm2d(4 * num_features * num_attentions)
        self.bn_last = nn.BatchNorm2d(num_features * num_attentions)

    def forward(self,feature_maps,attention_reserve=(1,1)):
        B,C,H,W=feature_maps.shape
        if type(attention_reserve) == tuple:
            attention_size = (int(H * attention_reserve[0]), int(W * attention_maps[1]))
        else:
            attention_size = (attention_reserve.shape[2], attention_reserve.shape[3])
        feature_maps = self.conv_extract(feature_maps)
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)
        if feature_maps.size(2) > feature_maps_d.size(2):
            feature_maps = feature_maps - F.interpolate(feature_maps_d, (feature_maps.shape[2], feature_maps.shape[3]),
                                                        mode='nearest')
        attention_maps = (
            torch.tanh(F.interpolate(attention_reserve.detach(), (H, W), mode='bilinear', align_corners=True))).unsqueeze(
            2) if type(attention_reserve) != tuple else 1
        feature_maps = feature_maps.unsqueeze(1)
        feature_maps = (feature_maps * attention_maps).reshape(B, -1, H, W)
        feature_maps0 = self.conv0(feature_maps)


class AttentionMap(nn.Module):
    def __init__(self,in_channels,out_channels,reserve_filter=1):
        super().__init__()
        self.register_buffer('mask', torch.zeros([1, 1, 24, 24]))
        self.num_attentions = out_channels
        self.mask[0, 0, 2:-2, 2:-2] = 1
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                      padding=1)  # extracting feature map from backbone
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.reserve_filter=reserve_filter
    def forward(self,x):
        if self.num_attentions==0:
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        x=self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        mask = F.interpolate(self.mask, (x.shape[2], x.shape[3]), mode='nearest')
        x_bn=x
        # 使用 torch.where 来保留大于1的值，其余置为0
        x_filtered = torch.where(x_bn > self.reserve_filter, x_bn, torch.zeros_like(x_bn))
        return x * mask,x_bn*x



class RM(nn.Module):
    def __init__(self,net='xception',feature_layer='b3',attention_layer='final',num_classes=2,
                 pretrained=False, M=8,size=(380,380),dropout_rate=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.net=xception(num_classes)
        self.feature_layer=feature_layer
        self.M=M
        self.attn_layer=attention_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1, 3, size[0], size[1]))
        self.attentions=AttentionMap(layers[self.attn_layer].shape[1], self.M)
        if pretrained:
            a=torch.load(pretrained,map_location='cpu')
            keys={i:a['state_dict'][i] for i in a.keys() if i.startswith('net')}
            if not keys:
                keys=a['state_dict']
            self.net.load_state_dict(keys,strict=False)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)

    def forward(self):
        pass

    def train_ep(self,x):
        layers=self.net(x)
        feature_maps=layers[self.feature_layer]
        raw_attentions=layers[self.attn_layer]
        attention_maps,attention_reserve=self.attentions(raw_attentions)
        dropout_mask = self.dropout(torch.ones([attention_maps.shape[0], self.M, 1], device=x.device))
        attention_maps=attention_maps*torch.unsqueeze(dropout_mask,-1)
        dropout_mask2 = self.dropout(torch.ones([attention_reserve.shape[0], self.M, 1], device=x.device))
        attention_reserve = attention_reserve * torch.unsqueeze(dropout_mask, -1)




