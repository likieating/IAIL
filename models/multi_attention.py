import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from convs.xception_dilated import xception
from convs.efficientnet import EfficientNet
import kornia
import torchvision.models as torchm

def cont_grad(x,rate=1):
    return rate*x+(1-rate)*x.detach()
def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]



class AttentionMap(nn.Module):

    def __init__(self,device,in_channels, out_channels):
        super(AttentionMap, self).__init__()
        self.device=device
        self.register_buffer('mask', torch.zeros([1, 1, 24, 24]))
        self.mask[0, 0, 2:-2, 2:-2] = 1
        self.num_attentions = out_channels
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                      padding=1)  # extracting feature map from backbone
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.num_attentions == 0:
            return torch.ones([x.shape[0], 1, 1, 1], device=self.device)
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x) + 1
        mask = F.interpolate(self.mask, (x.shape[2], x.shape[3]), mode='nearest')
        return x * mask


class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, attentions, norm=2):
        H, W = features.size()[-2:]
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(attentions, size=(H, W), mode='bilinear', align_corners=True)
        if norm == 1:
            attentions = attentions + 1e-8
        if len(features.shape) == 4:
            feature_matrix = torch.einsum('imjk,injk->imn', attentions, features)
        else:
            feature_matrix = torch.einsum('imjk,imnjk->imn', attentions, features)
        if norm == 1:
            w = torch.sum(attentions, dim=(2, 3)).unsqueeze(-1)
            feature_matrix /= w
        if norm == 2:
            feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)
        if norm == 3:
            w = torch.sum(attentions, dim=(2, 3)).unsqueeze(-1) + 1e-8
            feature_matrix /= w
        return feature_matrix


class Texture_Enhance_v1(nn.Module):
    def __init__(self, num_features, num_attentions):
        super().__init__()
        # self.output_features=num_features
        self.output_features = num_features * 4
        self.output_features_d = num_features
        self.conv0 = nn.Conv2d(num_features, num_features, 1)
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features * 2, num_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * num_features)
        self.conv3 = nn.Conv2d(num_features * 3, num_features, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(3 * num_features)
        self.conv_last = nn.Conv2d(num_features * 4, num_features * 4, 1)
        self.bn4 = nn.BatchNorm2d(4 * num_features)
        self.bn_last = nn.BatchNorm2d(num_features * 4)

    def forward(self, feature_maps, attention_maps=(1, 1)):
        B, N, H, W = feature_maps.shape
        if type(attention_maps) == tuple:
            attention_size = (int(H * attention_maps[0]), int(W * attention_maps[1]))
        else:
            attention_size = (attention_maps.shape[2], attention_maps.shape[3])
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)
        feature_maps = feature_maps - F.interpolate(feature_maps_d, (feature_maps.shape[2], feature_maps.shape[3]),
                                                    mode='nearest')
        feature_maps0 = self.conv0(feature_maps)
        feature_maps1 = self.conv1(F.relu(self.bn1(feature_maps0), inplace=True))
        feature_maps1_ = torch.cat([feature_maps0, feature_maps1], dim=1)
        feature_maps2 = self.conv2(F.relu(self.bn2(feature_maps1_), inplace=True))
        feature_maps2_ = torch.cat([feature_maps1_, feature_maps2], dim=1)
        feature_maps3 = self.conv3(F.relu(self.bn3(feature_maps2_), inplace=True))
        feature_maps3_ = torch.cat([feature_maps2_, feature_maps3], dim=1)
        feature_maps = self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_), inplace=True)))
        return feature_maps, feature_maps_d

class USKDLoss(nn.Module):
    """ PyTorch version of USKD """

    def __init__(self,device,
                 channel=728,
                 alpha=1.0,
                 beta=0.1,
                 mu=0.005,
                 num_classes=2,
                 ):
        super(USKDLoss, self).__init__()
        self.device=device
        self.channel = channel
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.fc = nn.Linear(channel, num_classes)
        self.zipf = 1 / torch.arange(1, num_classes + 1).to(self.device)

    def forward(self, fea_mid, logit_s, gt_label):

        if len(gt_label.size()) > 1:
            value, label = torch.sort(gt_label, descending=True, dim=-1)
            value = value[:, :2]
            label = label[:, :2]
        else:
            label = gt_label.view(len(gt_label), 1)
            value = torch.ones_like(label)

        N, c = logit_s.shape

        # final logit
        s_i = F.softmax(logit_s, dim=1)
        s_t = torch.gather(s_i, 1, label)

        # soft target label
        p_t = s_t ** 2
        p_t = p_t + value - p_t.mean(0, True)
        p_t[value == 0] = 0
        p_t = p_t.detach()

        s_i = self.log_softmax(logit_s)
        s_t = torch.gather(s_i, 1, label)
        loss_t = - (p_t * s_t).sum(dim=1).mean()

        # weak supervision
        if len(gt_label.size()) > 1:
            target = gt_label * 0.9 + 0.1 * torch.ones_like(logit_s) / c
        else:
            target = torch.zeros_like(logit_s).scatter_(1, label, 0.9) + 0.1 * torch.ones_like(logit_s) / c

        # weak logit
        w_fc = self.fc(fea_mid)
        w_i = self.log_softmax(w_fc)
        loss_weak = - (self.mu * target * w_i).sum(dim=1).mean()

        # N*class
        w_i = F.softmax(w_fc, dim=1)
        w_t = torch.gather(w_i, 1, label)

        # rank
        rank = w_i / (1 - w_t.sum(1, True) + 1e-6) + s_i / (1 - s_t.sum(1, True) + 1e-6)

        # soft non-target labels
        _, rank = torch.sort(rank, descending=True, dim=-1)
        z_i = self.zipf.repeat(N, 1)
        ids_sort = torch.argsort(rank)
        z_i = torch.gather(z_i, dim=1, index=ids_sort)

        mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()

        logit_s = logit_s[mask].reshape(N, -1)
        ns_i = self.log_softmax(logit_s)

        nz_i = z_i[mask].reshape(N, -1)
        nz_i = nz_i / nz_i.sum(dim=1, keepdim=True)

        nz_i = nz_i.detach()
        loss_non = - (nz_i * ns_i).sum(dim=1).mean()

        # overall
        loss_uskd = self.alpha * loss_t + self.beta * loss_non + loss_weak

        return loss_uskd



class compute_loss(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.USKDloss=USKDLoss(device)
    def forward(self,layers,y,mid_fea='b6'):
        fea_mid = layers[mid_fea]
        B,C,H,W=fea_mid.shape
        gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        fea_mid = gap(fea_mid)
        x_flat = fea_mid.view(B, C)
        loss_uskd = self.USKDloss(x_flat,layers['logits'], y)
        return loss_uskd



class Texture_Enhance_v2(nn.Module):
    def __init__(self, num_features, num_attentions):
        super().__init__()
        self.output_features = num_features
        self.output_features_d = num_features
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

        self.M = num_attentions

    def cat(self, a, b):
        B, C, H, W = a.shape
        c = torch.cat([a.reshape(B, self.M, -1, H, W), b.reshape(B, self.M, -1, H, W)], dim=2).reshape(B, -1, H, W)
        return c

    def forward(self, feature_maps, attention_maps=(1, 1)):
        B, N, H, W = feature_maps.shape
        if type(attention_maps) == tuple:
            attention_size = (int(H * attention_maps[0]), int(W * attention_maps[1]))
        else:
            attention_size = (attention_maps.shape[2], attention_maps.shape[3])
        feature_maps = self.conv_extract(feature_maps)
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)
        if feature_maps.size(2) > feature_maps_d.size(2):
            feature_maps = feature_maps - F.interpolate(feature_maps_d, (feature_maps.shape[2], feature_maps.shape[3]),
                                                        mode='nearest')
        attention_maps = (
            torch.tanh(F.interpolate(attention_maps.detach(), (H, W), mode='bilinear', align_corners=True))).unsqueeze(
            2) if type(attention_maps) != tuple else 1
        feature_maps = feature_maps.unsqueeze(1)
        feature_maps = (feature_maps * attention_maps).reshape(B, -1, H, W)
        feature_maps0 = self.conv0(feature_maps)
        feature_maps1 = self.conv1(F.relu(self.bn1(feature_maps0), inplace=True))
        feature_maps1_ = self.cat(feature_maps0, feature_maps1)
        feature_maps2 = self.conv2(F.relu(self.bn2(feature_maps1_), inplace=True))
        feature_maps2_ = self.cat(feature_maps1_, feature_maps2)
        feature_maps3 = self.conv3(F.relu(self.bn3(feature_maps2_), inplace=True))
        feature_maps3_ = self.cat(feature_maps2_, feature_maps3)
        feature_maps = F.relu(self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_), inplace=True))),
                              inplace=True)
        feature_maps = feature_maps.reshape(B, -1, N, H, W)
        return feature_maps, feature_maps_d


class Auxiliary_Loss_v2(nn.Module):
    def __init__(self, M, N, C, alpha=0.05, margin=1, inner_margin=[0.1, 5]):
        super().__init__()
        self.register_buffer('feature_centers', torch.zeros(M, N))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.num_classes = C
        self.margin = margin
        self.atp = AttentionPooling()
        self.register_buffer('inner_margin', torch.Tensor(inner_margin))

    def forward(self, feature_map_d, attentions, y):
        B, N, H, W = feature_map_d.size()
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(attentions, (H, W), mode='bilinear', align_corners=True)
        feature_matrix = self.atp(feature_map_d, attentions)
        feature_centers = self.feature_centers
        center_momentum = feature_matrix - feature_centers
        real_mask = (y == 0).view(-1, 1, 1)
        fcts = self.alpha * torch.mean(center_momentum * real_mask, dim=0) + feature_centers
        fctsd = fcts.detach()
        if self.training:
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(fctsd, torch.distributed.ReduceOp.SUM)
                    fctsd /= torch.distributed.get_world_size()
                self.feature_centers = fctsd
        inner_margin = self.inner_margin[y]
        intra_class_loss = F.relu(
            torch.norm(feature_matrix - fcts, dim=[1, 2]) * torch.sign(inner_margin) - inner_margin)
        intra_class_loss = torch.mean(intra_class_loss)
        inter_class_loss = 0
        for j in range(M):
            for k in range(j + 1, M):
                inter_class_loss += F.relu(self.margin - torch.dist(fcts[j], fcts[k]), inplace=False)
        inter_class_loss = inter_class_loss / M / self.alpha
        # fmd=attentions.flatten(2)
        # diverse_loss=torch.mean(F.relu(F.cosine_similarity(fmd.unsqueeze(1),fmd.unsqueeze(2),dim=3)-self.margin,inplace=True)*(1-torch.eye(M,device=attentions.device)))
        return intra_class_loss + inter_class_loss, feature_matrix


class Auxiliary_Loss_v1(nn.Module):
    def __init__(self, M, N, C, alpha=0.05, margin=1, inner_margin=[0.01, 0.02]):
        super().__init__()
        self.register_buffer('feature_centers', torch.zeros(M, N))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.num_classes = C
        self.margin = margin
        self.atp = AttentionPooling()
        self.register_buffer('inner_margin', torch.Tensor(inner_margin))

    def forward(self, feature_map_d, attentions, y):
        B, N, H, W = feature_map_d.size()
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(attentions, (H, W), mode='bilinear', align_corners=True)
        feature_matrix = self.atp(feature_map_d, attentions)
        feature_centers = self.feature_centers.detach()
        center_momentum = feature_matrix - feature_centers
        fcts = self.alpha * torch.mean(center_momentum, dim=0) + feature_centers
        fctsd = fcts.detach()
        if self.training:
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(fctsd, torch.distributed.ReduceOp.SUM)
                    fctsd /= torch.distributed.get_world_size()
                self.feature_centers = fctsd
        inner_margin = torch.gather(self.inner_margin.repeat(B, 1), 1, y.unsqueeze(1))
        intra_class_loss = F.relu(torch.norm(feature_matrix - fcts, dim=-1) - inner_margin)
        intra_class_loss = torch.mean(intra_class_loss)
        inter_class_loss = 0
        for j in range(M):
            for k in range(j + 1, M):
                inter_class_loss += F.relu(self.margin - torch.dist(fcts[j], fcts[k]), inplace=False)
        inter_calss_loss = inter_class_loss / M / self.alpha
        # fmd=attentions.flatten(2)
        # inter_class_loss=torch.mean(F.relu(F.cosine_similarity(fmd.unsqueeze(1),fmd.unsqueeze(2),dim=3)-self.margin,inplace=True)*(1-torch.eye(M,device=attentions.device)))
        return intra_class_loss + inter_class_loss, feature_matrix


class MAT(nn.Module):
    def __init__(self, device=torch.device("cuda:3"),net='xception', feature_layer='b3', attention_layer='final', num_classes=2, M=8, mid_dims=256, \
                 dropout_rate=0.5, drop_final_rate=0.5, pretrained=False, alpha=0.05, size=(299,299), margin=1,
                 inner_margin=[0.01, 0.02]):
        super(MAT, self).__init__()

        self.num_classes = num_classes
        self.M = M
        if 'xception' in net:
            self.net = xception(num_classes)
        elif net.split('-')[0] == 'efficientnet':
            self.net = EfficientNet.from_pretrained(net, advprop=True, num_classes=num_classes)
        self.feature_layer = feature_layer
        self.attention_layer = attention_layer
        self.device=device
        with torch.no_grad():
            layers = self.net(torch.zeros(1, 3, size[0], size[1]))
        num_features = layers[self.feature_layer].shape[1]
        self.mid_dims = mid_dims
        if pretrained:
            a = torch.load(pretrained, map_location='cpu')
            keys = {i: a['state_dict'][i] for i in a.keys() if i.startswith('net')}
            if not keys:
                keys = a['state_dict']
            self.net.load_state_dict(keys, strict=False)
        self.attentions = AttentionMap(self.device,layers[self.attention_layer].shape[1], self.M)
        self.atp = AttentionPooling()
        # self.texture_enhance = Texture_Enhance_v2(num_features, M)
        # self.num_features = self.texture_enhance.output_features
        # self.num_features_d = self.texture_enhance.output_features_d
        # self.projection_local = nn.Sequential(nn.Linear(M * self.num_features, mid_dims), nn.Hardswish(),
        #                                       nn.Linear(mid_dims, mid_dims))
        # self.project_final = nn.Linear(layers['final'].shape[1], mid_dims)
        # self.ensemble_classifier_fc = nn.Sequential(nn.Linear(mid_dims * 2, mid_dims), nn.Hardswish(),
        #                                             nn.Linear(mid_dims, num_classes))
        # self.auxiliary_loss = Auxiliary_Loss_v2(M, self.num_features_d, num_classes, alpha, margin, inner_margin)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)
        # self.dropout_final = nn.Dropout(drop_final_rate, inplace=True)
        self.compute_loss=compute_loss(self.device)
        # self.center_loss=Center_Loss(self.num_features*M,num_classes)

    def train_batch(self, x, y,old_net=None, jump_aux=False, drop_final=False,init=False):
        layers = self.net(x)

        if self.feature_layer == 'logits':
            logits = layers['logits']
            loss = F.cross_entropy(logits, y)
            return dict(loss=loss, logits=logits)
        logits_new=layers['logits']
        if old_net is not None:
            layers_old = old_net(x)
            logits_old = layers_old['logits']
            feature_maps = layers_old[self.feature_layer]
        else:
            feature_maps = layers[self.feature_layer]
        raw_attentions = layers[self.attention_layer]
        attention_maps_ = self.attentions(raw_attentions)
        dropout_mask = self.dropout(torch.ones([attention_maps_.shape[0], self.M, 1], device=self.device))
        attention_maps = attention_maps_ * torch.unsqueeze(dropout_mask, -1)
        # feature_maps, feature_maps_d = self.texture_enhance(feature_maps, attention_maps_)
        # feature_maps_d = feature_maps_d - feature_maps_d.mean(dim=[2, 3], keepdim=True)
        # feature_maps_d = feature_maps_d / (torch.std(feature_maps_d, dim=[2, 3], keepdim=True) + 1e-8)
        # feature_matrix_ = self.atp(feature_maps, attention_maps_)
        # feature_matrix = feature_matrix_ * dropout_mask

        # B, M, N = feature_matrix.size()
        # if not jump_aux:
        #     aux_loss, feature_matrix_d = self.auxiliary_loss(feature_maps_d, attention_maps_, y)
        # else:
        #     feature_matrix_d = self.atp(feature_maps_d, attention_maps_)
        #     aux_loss = 0
        # feature_matrix = feature_matrix.view(B, -1)
        # feature_matrix = F.hardswish(self.projection_local(feature_matrix))
        # final = layers['final']
        # attention_maps = attention_maps.sum(dim=1, keepdim=True)
        return layers,attention_maps
        # final = self.atp(final, attention_maps, norm=1).squeeze(1)
        # final = self.dropout_final(final)
        # projected_final = F.hardswish(self.project_final(final))
        # # projected_final=self.dropout(projected_final.view(B,1,-1)).view(B,-1)
        # if drop_final:
        #     projected_final *= 0
        # feature_matrix = torch.cat((feature_matrix, projected_final), 1)
        # ensemble_logit = self.ensemble_classifier_fc(feature_matrix)
        # ensemble_loss = F.cross_entropy(ensemble_logit, y)
        # if old_net is not None:
        #     kd_loss = _KD_loss(logits_new, logits_old, 20)
        #     return dict(ensemble_loss=ensemble_loss, aux_loss=aux_loss, attention_maps=attention_maps_,
        #                 ensemble_logit=ensemble_logit, feature_matrix=feature_matrix_,
        #                 feature_matrix_d=feature_matrix_d, kd_loss=kd_loss), logits_new
        # else:
        #     return dict(ensemble_loss=ensemble_loss, aux_loss=aux_loss, attention_maps=attention_maps_,
        #                 ensemble_logit=ensemble_logit, feature_matrix=feature_matrix_,
        #                 feature_matrix_d=feature_matrix_d), logits_new

    def forward(self, x, y=0,old_net=None, train_batch=False, AG=None,init=False):
        if train_batch:
            if AG is None:
                return self.train_batch(x, y)
            else:
                # if old_net is not None:
                #     with torch.no_grad():
                #         loss_pack=old_net(x,y,train_batch=True,AG=None)
                #     old_logits=loss_pack[0]['ensemble_logit']
                #     loss_pack=loss_pack[0]
                #     attn_maps=loss_pack['attention_maps']
                # else:
                # loss_pack,logits1 = self.train_batch(x, y,init=False)
                # attn_maps=loss_pack['attention_maps']
                layers,attn_maps=self.train_batch(x, y,init=False)
                with torch.no_grad():
                    Xaug, index = AG.agda(x, attn_maps)
                # self.eval()
                layers,attn_maps = self.train_batch(Xaug, y, jump_aux=False)
                logits=layers['logits']
                features=layers['features']
                loss=self.compute_loss(layers,y)


                return loss,logits,features
        layers = self.net(x)
        logits = layers['logits']
        return logits
        # raw_attentions = layers[self.attention_layer]
        # attention_maps = self.attentions(raw_attentions)
        # feature_maps = layers[self.feature_layer]
        # feature_maps, feature_maps_d = self.texture_enhance(feature_maps, attention_maps)
        # feature_matrix = self.atp(feature_maps, attention_maps)
        # B, M, N = feature_matrix.size()
        # feature_matrix = self.dropout(feature_matrix)
        # feature_matrix = feature_matrix.view(B, -1)
        # feature_matrix = F.hardswish(self.projection_local(feature_matrix))
        # final = layers['final']
        # attention_maps2 = attention_maps.sum(dim=1, keepdim=True)
        # final = self.atp(final, attention_maps2, norm=1).squeeze(1)
        # projected_final = F.hardswish(self.project_final(final))
        # feature_matrix = torch.cat((feature_matrix, projected_final), 1)
        # ensemble_logit = self.ensemble_classifier_fc(feature_matrix)
        # return ensemble_logit


def load_state(net, ckpt):
    sd = net.state_dict()
    nd = {}
    for i in ckpt:
        if i in sd and sd[i].shape == ckpt[i].shape:
            nd[i] = ckpt[i]
    net.load_state_dict(nd, strict=False)


class netrunc(nn.Module):
    def __init__(self, net='xception', feature_layer='b3', num_classes=2, dropout_rate=0.5, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        if 'xception' in net:
            self.net = xception(num_classes, escape=feature_layer)
        elif net.split('-')[0] == 'efficientnet':
            self.net = EfficientNet.from_pretrained(net, advprop=True, num_classes=num_classes, escape=feature_layer)
        self.feature_layer = feature_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1, 3, 100, 100))
        num_features = layers[self.feature_layer].shape[1]
        if pretrained:
            a = torch.load(pretrained, map_location='cpu')
            keys = {i: a['state_dict'][i] for i in a.keys() if i.startswith('net')}
            if not keys:
                keys = a['state_dict']
            load_state(self.net, keys)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.texture_enhance = Texture_Enhance_v2(num_features, 1)
        self.num_features = self.texture_enhance.output_features
        self.fc = nn.Linear(self.num_features, self.num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        layers = self.net(x)
        feature_maps = layers[self.feature_layer]
        feature_maps, _ = self.texture_enhance(feature_maps, (0.2, 0.2))
        x = self.pooling(feature_maps)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
