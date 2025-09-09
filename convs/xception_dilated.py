import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class xception(nn.Module):
    def __init__(self,num_classes=2,pretrained='imagenet',escape='',isall='yes'):
        super(xception, self).__init__()
        self.escape=escape
        self.isall=isall
        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)
        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.dilated_conv1 = nn.Conv2d(2048,2048, kernel_size=3, dilation=1, padding=1)
        self.dilated_conv2 = nn.Conv2d(2048,2048, kernel_size=3, dilation=2, padding=2)
        self.dilated_conv3 = nn.Conv2d(2048,2048, kernel_size=3, dilation=4, padding=4)
        self.conv_fusion = nn.Conv2d( 2048* 3, 2048, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4=nn.ReLU(inplace=True)
        self.last_linear = nn.Linear(2048, num_classes)
        self.seq=[]
        self.seq.append(('b0',[self.conv1,lambda x:self.bn1(x),self.relu1,self.conv2,lambda x:self.bn2(x)]))
        self.seq.append(('b1',[self.relu2,self.block1]))
        self.seq.append(('b2',[self.block2]))
        self.seq.append(('b3',[self.block3]))
        self.seq.append(('b4',[self.block4]))
        self.seq.append(('b5',[self.block5]))
        self.seq.append(('b6',[self.block6]))
        self.seq.append(('b7',[self.block7]))
        self.seq.append(('b8',[self.block8]))
        self.seq.append(('b9',[self.block9]))
        self.seq.append(('b10',[self.block10]))
        self.seq.append(('b11',[self.block11]))
        self.seq.append(('b12',[self.block12]))
        self.seq.append(('final',[self.conv3,lambda x:self.bn3(x),self.relu3,self.conv4,lambda x:self.bn4(x)]))
        self.seq.append(('features',[self.relu4,lambda x:F.adaptive_avg_pool2d(x, (1, 1)),lambda x:x.view(x.size(0), -1)]))
        self.seq.append(('logits',[self.last_linear]))
        if pretrained=='imagenet':
            model_url = '/home/tangshuai/dmp-bk/pre_weights/xception-43020ad28.pth'
            pretrained_weights = torch.load(model_url)  # 直接加载模型权重
            self.load_state_dict(pretrained_weights, strict=False)
        elif pretrained:
            ckpt=torch.load(pretrained,map_location='cpu')
            self.load_state_dict(ckpt['state_dict'])
        else:
            #------- init weights --------
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            #-----------------------------

        self.seq2 = []
        self.seq2.append(('b4', [self.block4]))
        self.seq2.append(('b5', [self.block5]))
        self.seq2.append(('b6', [self.block6]))
        self.seq2.append(('b7', [self.block7]))
        self.seq2.append(('b8', [self.block8]))
        self.seq2.append(('b9', [self.block9]))
        self.seq2.append(('b10', [self.block10]))
        self.seq2.append(('b11', [self.block11]))
        self.seq2.append(('b12', [self.block12]))
        self.seq2.append(('final', [self.conv3, lambda x: self.bn3(x), self.relu3, self.conv4, lambda x: self.bn4(x)]))
        self.seq2.append(
            ('features', [self.relu4, lambda x: F.adaptive_avg_pool2d(x, (1, 1)), lambda x: x.view(x.size(0), -1)]))
        self.seq2.append(('logits', [self.last_linear]))

    def forward(self, x,fea_layer=None,isall='yes',sta=0):
        if isall == 'yes':
            layers={}
            for stage in self.seq:
                for f in stage[1]:
                    x=f(x)
                layers[stage[0]]=x
                if stage[0]==self.escape:
                    break
            return layers
        elif isall =='fea':
            if sta==0:
                for stage in self.seq:
                    for f in stage[1]:
                        x=f(x)
                    if stage[0]==fea_layer:
                        break
                return x
            else:
                layers = {}
                flag=0
                for stage in self.seq:
                    if stage[0]==fea_layer:
                        flag=1
                    if flag != 1  or stage[0]==fea_layer:
                        continue
                    for f in stage[1]:
                        x=f(x)
                    layers[stage[0]] = x
                    if stage[0] == self.escape:
                        break
                return layers



