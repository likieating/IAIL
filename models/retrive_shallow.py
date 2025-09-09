import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from convs.xception_rm import xception
import os
from torchvision.ops import roi_align
import pandas as pd
# upsampling version
class AttentionMap(nn.Module):
    def __init__(self,in_channels,out_channels,reserve_filter=1,up_func="BI",
                 window_size=(4,4),threshold_nms=0.5):
        super().__init__()
        self.register_buffer('mask', torch.zeros([1, 1, 24, 24]))
        self.num_attentions = out_channels
        self.mask[0, 0, 2:-2, 2:-2] = 1
        self.up_func=up_func
        self.conv_transpose = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                      padding=1)  # extracting feature map from backbone
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.reserve_filter=reserve_filter
        self.window,_=window_size



    def forward(self,x):
        """
        Args:
            x:xception_features:(2048,320/16,320/16)
        return:
            attn maps:
            (B,C,H*2,W*2)
        """
        if self.num_attentions==0:
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)

        if self.up_func=='BI':
            x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        elif self.up_func=='DeConv':
            x=self.conv_transpose(x)
        x=self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        mask = F.interpolate(self.mask, (x.shape[2], x.shape[3]), mode='nearest')
        x_bn=x
        # 使用 torch.where 来保留大于reserve_filter的值，其余置为0
        #b num 40 40
        x_filtered = torch.where(x_bn > self.reserve_filter, x_bn, torch.zeros_like(x_bn))
        imp_attns,imp_loca=NMS(x_filtered,self.num_attentions,kernel_size=self.window)
        return imp_attns,imp_loca

#TODO:补一下
def sliding_window_sum(x, kernel_size=3, stride=1, padding=2):
    """
    对输入张量 x 应用滑动窗口并对每个窗口内的元素求和。

    参数:
        x (torch.Tensor): 输入张量，形状为 (B, C, H, W)
        kernel_size (int): 滑动窗口的大小，默认为 3
        stride (int): 滑动窗口的步幅，默认为 1
        padding (int): 输入的零填充，默认为 1

    返回:
        window_sum (torch.Tensor): 每个窗口内元素的和，形状为 (B, C, H, W)
    """
    B, C, H, W = x.shape
    # 使用 unfold 提取滑动窗口
    # 输出形状为 (B, C * kernel_size * kernel_size, L)，其中 L = H * W
    windows = F.unfold(x, kernel_size=kernel_size, stride=stride, padding=padding)

    # 计算每个窗口内元素的和
    # 调整形状为 (B, C, kernel_size * kernel_size, H * W)
    H=H-kernel_size+2*padding+1
    W=H
    windows = windows.view(B, C, kernel_size * kernel_size, H * W)

    # 对 kernel_size * kernel_size 维度求和，得到 (B, C, H * W)
    window_sum = windows.sum(dim=2)

    # 重新调整为 (B, C, H, W)
    window_sum = window_sum.view(B, C, H, W)

    return window_sum


def non_maximum_suppression(window_sum, num, kernel_size=3):
    """
    对窗口和应用非极大值抑制，并选择前 num 个窗口。

    参数:
        window_sum (torch.Tensor): 窗口和，形状为 (B, C, H, W)
        num (int): 每个 (B, C) 选择的窗口数量
        kernel_size (int): NMS 的邻域大小，默认为 3

    返回:
        top_values (torch.Tensor): 前 num 个窗口的和，形状为 (B, C, num)
        top_indices (torch.Tensor): 前 num 个窗口的索引，形状为 (B, C, num)
    """
    B, C, H, W = window_sum.shape

    # 使用最大池化找到局部最大值
    # max_pool 输出的形状为 (B, C, H, W)
    max_pooled = F.max_pool2d(window_sum, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    # 创建掩码，标记出局部最大值的位置
    mask = (window_sum == max_pooled).float()

    # 将非局部最大值的位置置零
    masked_window_sum = window_sum * mask

    # 将窗口和展开为 (B, C, H*W)
    masked_window_sum_flat = masked_window_sum.view(B, C, -1)

    # 使用 topk 选择每个 (B, C) 的前 num 个窗口
    top_values, top_indices = torch.topk(masked_window_sum_flat, num, dim=2)

    return top_values, top_indices


def extract_top_windows(x, top_indices, num, kernel_size=3, stride=1, padding=2):
    """
    根据 top_indices 从原始张量中提取窗口内容。

    参数:
        x (torch.Tensor): 原始输入张量，形状为 (B, C, H, W)
        top_indices (torch.Tensor): 前 num 个窗口的索引，形状为 (B, C, num)
        num (int): 每个 (B, C) 选择的窗口数量
        kernel_size (int): 窗口大小，默认为 3
        stride (int): 滑动窗口的步幅，默认为 1
        padding (int): 输入的零填充，默认为 1

    返回:
        top_windows (torch.Tensor): 选中的窗口内容，形状为 (B, C, num, kernel_size, kernel_size)
        positions (torch.Tensor): 选中窗口的位置坐标，形状为 (B, C, num, 4)
    """
    B, C, H, W = x.shape

    # 使用 unfold 提取所有滑动窗口
    # 输出形状为 (B, C * kernel_size * kernel_size, L)，其中 L = H * W
    windows = F.unfold(x, kernel_size=kernel_size, stride=stride, padding=padding)

    # 重新调整形状为 (B, C, kernel_size * kernel_size, H * W)
    windows = windows.view(B, C, kernel_size * kernel_size, H * W)

    # 根据 top_indices 选择对应的窗口
    # top_indices 的形状为 (B, C, num)
    # 需要在最后一个维度 (H * W) 上进行索引
    top_windows = torch.gather(windows, dim=3,
                               index=top_indices.unsqueeze(2).expand(-1, -1, kernel_size * kernel_size, -1))

    # 重新调整形状为 (B, C, num, kernel_size * kernel_size)
    top_windows = top_windows.permute(0, 1, 3, 2).contiguous()
    top_windows = top_windows.view(B, C, num, kernel_size, kernel_size)

    # 计算每个索引对应的窗口中心位置 (h, w)
    top_indices_h = top_indices // W
    top_indices_w = top_indices % W

    # 计算窗口的左上角和右下角坐标
    # 假设窗口大小为 3x3，步幅为 1，填充为 1
    # x1 = w - 1, y1 = h - 1, x2 = w + 1, y2 = h + 1
    x1 = (top_indices_w - kernel_size // 2).clamp(min=0, max=W - 1)
    y1 = (top_indices_h - kernel_size // 2).clamp(min=0, max=H - 1)
    x2 = (top_indices_w + kernel_size // 2).clamp(min=0, max=W - 1)
    y2 = (top_indices_h + kernel_size // 2).clamp(min=0, max=H - 1)

    # 组合坐标
    positions = torch.stack([x1, y1, x2, y2], dim=3)  # (B, C, num, 4)

    return top_windows, positions


def NMS(x, num, kernel_size=3, stride=1, padding=2):
    """
    完整的处理流程：滑动窗口求和、NMS、选择 top num 窗口、提取窗口内容和位置信息。

    参数:
        x (torch.Tensor): 输入张量，形状为 (B, C, H, W)
        num (int): 每个 (B, C) 选择的窗口数量
        kernel_size (int): 滑动窗口的大小，默认为 3
        stride (int): 滑动窗口的步幅，默认为 1
        padding (int): 输入的零填充，默认为 1

    返回:
        top_windows (torch.Tensor): 选中的窗口内容，形状为 (B, C, num, kernel_size, kernel_size)
        positions (torch.Tensor): 选中窗口的位置坐标，形状为 (B, C, num, 4)
    """
    # 步骤 1: 滑动窗口求和
    window_sum = sliding_window_sum(x, kernel_size=kernel_size, stride=stride, padding=padding)  # (B, C, H, W)

    # 步骤 2: 应用 NMS 并选择前 num 个窗口
    top_values, top_indices = non_maximum_suppression(window_sum, num, kernel_size=kernel_size)

    # 步骤 3: 提取选中的窗口内容和位置信息
    top_windows, positions = extract_top_windows(x, top_indices, num, kernel_size=kernel_size, stride=stride,
                                                 padding=padding)

    return top_windows, positions

class shallow(nn.Module):
    def __init__(self,input_chans,fea_chans=8,num=10,winodow_size=(3,3)):
        super().__init__()
        self.xx,self.yy=winodow_size
        self.input_chans=input_chans
        self.num=num
        self.fea_chans=fea_chans
        self.conv1=nn.Conv2d(input_chans,input_chans,kernel_size=3,
                                      padding=1)
        self.conv2=nn.Conv2d(input_chans,fea_chans,kernel_size=1,bias=False)
    def forward(self,features,imp_attns,imp_locat):
        """
        Args:
            features: shallow features (B,C,H,W)
            imp_attns: importent attns (B,C,N,window_x,window_y)
            imp_locat:important features' location(B,C,N,4)
        """
        features=self.conv1(features)
        features=self.conv2(features)
        #? 要做bn吗
        B, C, H, W = features.shape
        _, _, N, _ = imp_locat.shape
        window_x, window_y = self.xx,self.yy
        # 重塑 features 以处理每个通道的特征
        # 新的 batch size 是 B*C
        features_reshaped = features.view(B * C, 1, H, W)  # (B*C, 1, H, W)

        # 重塑 imp_locat 为 (B*C*N, 4)
        boxes = imp_locat.view(B * C * N, 4)  # (B*C*N, 4)

        # 创建批次索引
        batch_indices = torch.arange(B * C, device=features.device).repeat_interleave(N)  # (B*C*N,)
        # 组合批次索引和边界框，形成 (B*C*N, 5) 的张量
        boxes_with_batch = torch.cat([batch_indices.unsqueeze(1).float(), boxes], dim=1)  # (B*C*N, 5)

        # 使用 roi_align 提取窗口
        patches = roi_align(features_reshaped, boxes_with_batch, output_size=window_x, spatial_scale=1.0,
                            aligned=True)  # (B*C*N, 1, window_x, window_y)

        # 重塑为 (B, C, N, window_x, window_y)
        patches = patches.view(B, C, N, window_x, window_y)
        # B,C,N,window_x,window_y
        return patches*imp_attns,patches


class db_store(nn.Module):
    def __init__(self,window_size=3,sim_threshold=0.9):
        super().__init__()
        self.threhold=sim_threshold
        self.db_tmp=[]
        self.window_size=window_size
    def forward(self,patches,db):
        """
        Args:
            patches:b*c*num*window_size*window_size
            db:order*m*window_size*window_size
        """
        patches=patches.view(-1,self.window_size,self.window_size)
        if db.shape[0]==0:
            db=patches
            return db
        n=patches.shape[0]
        m=db.shape[0]
        patches_flat=patches.view(n,-1)
        db_flat=db.view(m,-1)
        dot_product = torch.matmul(patches_flat,db_flat.T)
        norm_n = torch.norm(patches_flat, p=2, dim=1, keepdim=True)  # 形状 (n, 1)
        norm_m = torch.norm(db_flat, p=2, dim=1, keepdim=True)  # 形状 (m, 1)
        #n m
        cosine_similarity = dot_product / (norm_n * norm_m.T)
        mask=(cosine_similarity<self.threhold).all(dim=1)
        mask_int=mask.int().cpu()
        df = pd.DataFrame(mask_int.numpy())  # 将Tensor转换为NumPy数组，再转换为DataFrame
        df.to_csv('mask_values.csv', index=False, header=False)
        ask_dim=mask.sum()
        filtered_n=patches[mask]
        db=db.view(-1,self.window_size,self.window_size)
        db=torch.cat((db,filtered_n),dim=0)
        return db

class db_retrive(nn.Module):
    def __init__(self,threshold=0.5,window_size=(3,3)):
        super().__init__()
    def forward(self,patches,db,db_weight):
        """
        Args:
            patches:b*c*n*window_size*window_size
            db:m*window_size*window_size
            db_weight(tensor):m
        """
        b,c,_,_,window=patches.shape
        patches = patches.view(-1, window, window)
        n = patches.shape[0]
        m = db.shape[0]
        patches_flat = patches.view(n, -1)
        db_flat = db.view(m, -1)
        dot_product = torch.matmul(patches_flat, db_flat.T)
        norm_n = torch.norm(patches_flat, p=2, dim=1, keepdim=True)  # 形状 (n, 1)
        norm_m = torch.norm(db_flat, p=2, dim=1, keepdim=True)  # 形状 (m, 1)
        # n m
        cosine_similarity = dot_product / (norm_n * norm_m.T)
        cosine_similarity=torch.clamp(cosine_similarity,min=0)
        norm_weights=db_weight[:m]/db_weight[:m].sum()
        norm_weights=norm_weights.unsqueeze(0).repeat(n,1)
        result=torch.mul(cosine_similarity,norm_weights)
        result=result.view(b,-1)
        result_sum=result.sum(dim=1)
        return result_sum



class RS_init(nn.Module):
    def __init__(self,db_fake,db_real,net,device,stage,feature_layer='b3',attention_layer='final',num_classes=2, M=8,size=(299,299),
                 window_size=(5,5),threshold_sim=0.8,threshold_pd=0.5,num_sources=4):
        super().__init__()
        self.net=net
        self.num_classes=num_classes
        self.M=M
        self.stage=stage
        self._device=device
        self.feature_layer=feature_layer
        self.attention_layer=attention_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1, 3, size[0], size[1]))
        num_features = layers[self.feature_layer].shape[1]
        self.attentions=AttentionMap(layers[self.attention_layer].shape[1],self.M,window_size=window_size).to(self._device)
        self.shallow=shallow(num_features,winodow_size=window_size).to(self._device)
        self.weight_dim=nn.Parameter(torch.randn(num_sources,1000)).to(self._device)
        self.db_fake=db_fake
        self.db_real=db_real
        self.window,_=window_size
        self.db_store=db_store(window_size=self.window,sim_threshold=threshold_sim).to(self._device)
        self.db_retrieve=db_retrive(threshold=threshold_pd,window_size=self.window).to(self._device)

    def forward(self,train_loader):
        with torch.no_grad():
            for i, (_, inputs, targets, order) in enumerate(train_loader):
                inputs, targets,order = inputs.to(self._device), targets.to(self._device),order.to(self._device)
                self.get_db(inputs,targets,order)
        print(1111)
        return self.db_fake,self.db_real
    def get_db(self,inputs,targets,order):
        """
        Args:
            inputs(tensor):b,c,299,299
            targets(tensor):b
            order(tensor):b
            attn_layer(tensor):b,c,h,w
        """
        targets_value=int(torch.unique(targets)[0].item())
        layers = self.net(inputs)
        attn_layer = layers[self.attention_layer]
        fea_layer = layers[self.feature_layer]
        imp_attns, imp_locat = self.attentions(attn_layer)
        # B,C,N,window_x,window_y
        extracted_weight, extracted = self.shallow(fea_layer, imp_attns, imp_locat)
        unique_value = int(torch.unique(order)[0].item())
        if targets_value==1:
            # #b n(logits)
            # result_sum=self.db_retrieve(extracted,self.db[order],self.weight_dim[order])
            self.db_fake[unique_value] = self.db_store(extracted, self.db_fake[unique_value])
        else:
            self.db_real[unique_value] = self.db_store(extracted, self.db_real[unique_value])



class contrastive_loss(nn.Module):
    def __init__(self,db_fake,db_real,net,device,stage,feature_layer='b3',attention_layer='final',num_classes=2, M=8,size=(299,299),
                 window_size=3,threshold_sim=0.8,threshold_pd=0.5,num_sources=4):
        super().__init__()
        self.net=net
        self.num_classes=num_classes
        self.M=M
        self.stage = stage
        self._device=device
        self.feature_layer=feature_layer
        self.attention_layer=attention_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1, 3, size[0], size[1]))
        num_features = layers[self.feature_layer].shape[1]
        self.attentions=AttentionMap(layers[self.attention_layer].shape[1],self.M).to(self._device)
        self.shallow=shallow(num_features).to(self._device)
        self.weight_dim=nn.Parameter(torch.randn(num_sources,1000)).to(self._device)
        self.db_fake=db_fake
        self.db_real=db_real
        self.db_store=db_store(window_size=window_size,sim_threshold=threshold_sim).to(self._device)
        self.db_retrieve=db_retrive(threshold=threshold_pd).to(self._device)
    def forward(self,fea_layer,attn_layer,targets,order):
        """
        Args:
            fea_layer: B,C,H,W
            attn_layer:B,C,H,W

        """

        mask=order!=self.order
        if torch.all(mask==False):
            return 0
        filtered_targets=targets[mask]
        filtered_order=order[mask]
        filtered_attn_layer=attn_layer[mask]
        filtered_fea_layer=fea_layer[mask]
        imp_attns, imp_locat = self.attentions(filtered_attn_layer)
        # B,C,N,window_x,window_y
        extracted_weight, extracted = self.shallow(filtered_fea_layer, imp_attns, imp_locat)
        m=filtered_order.shape[0]
        loss=0.0
        for i  in range(m):
            loss+=_compute_contrastive_loss(extracted[i],filtered_targets[i],self.db_real[filtered_order[i]],self.db_fake[filtered_order[i]])
        return loss/m


def _compute_contrastive_loss(inputs, input_labels, database_real, database_fake, margin=1.0):
    """
        计算对比损失（Contrastive Loss）

        参数:
        - inputs: Tensor of shape ( C,N,window_size,window_size)，输入特征向量
        - input_labels: Tensor of shape (1)
        - database_real: Tensor of shape (n_real,window_size,window_size)，真实人脸特征
        - database_fake: Tensor of shape (n_fake, window_size,window_size)，伪造人脸特征
        - margin: float，margin 超参数，默认为 1.0

        返回:
        - loss: scalar tensor，对比损失
        """
    batch_size, d = inputs.shape
    n_real = database_real.shape[0]
    n_fake = database_fake.shape[0]

    # 合并数据库特征和标签
    database = torch.cat([database_real, database_fake], dim=0)  # 形状: (n_real + n_fake, d)
    database_labels = torch.cat([torch.ones(n_real), torch.zeros(n_fake)], dim=0)  # 形状: (n_real + n_fake,)

    # 将输入特征扩展以匹配数据库大小
    # inputs: (batch_size, d) -> (batch_size, 1, d) -> (batch_size, n_db, d)
    inputs_expanded = inputs.unsqueeze(1).expand(-1, database.size(0), -1)  # (batch_size, n_db, d)
    # database: (n_db, d) -> (1, n_db, d) -> (batch_size, n_db, d)
    database_expanded = database.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, n_db, d)

    # 计算欧氏距离
    distances = torch.norm(inputs_expanded - database_expanded, p=2, dim=2)  # (batch_size, n_db)

    # 生成对比标签
    # input_labels: (batch_size,) -> (batch_size, 1)
    # database_labels: (n_db,) -> (1, n_db)
    # similarity: (batch_size, n_db)，1.0 表示相似，0.0 表示不相似
    similarity = (input_labels.unsqueeze(1) == database_labels.unsqueeze(0)).float()  # (batch_size, n_db)

    # 计算对比损失
    loss = torch.mean(
        similarity * (distances ** 2) +
        (1.0 - similarity) * torch.clamp(margin - distances, min=0.0) ** 2
    )

    return loss





class RS_eval(nn.Module):
    def __init__(self,db_real,db_fake,net,device,stage,log,feature_layer='b3',attention_layer='final',num_classes=2, M=8,size=(299,299),
                 window_size=3,threshold_sim=0.8,threshold_pd=0.5,num_sources=4):
        super().__init__()
        self.net = net
        self.num_classes = num_classes
        self.M = M
        self.stage = stage
        self.log=log
        self._device = device
        self.feature_layer = feature_layer
        self.attention_layer = attention_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1, 3, size[0], size[1]))
        num_features = layers[self.feature_layer].shape[1]
        self.attentions = AttentionMap(layers[self.attention_layer].shape[1], self.M).to(self._device)
        self.shallow = shallow(num_features).to(self._device)
        self.weight_dim =torch.ones(4,10000).to(self._device)
        self.db_fake = db_fake
        self.db_real = db_real
        self.db_store = db_store(window_size=window_size, sim_threshold=threshold_sim).to(self._device)
        self.db_retrieve = db_retrive(threshold=threshold_pd).to(self._device)

    def forward(self,eval_loader):
        y_order,y_pred_db_real,y_pred_db_fake, y_true = self.eval_cnn(eval_loader)
        # 如果是二维数据，可以将其保存为 DataFrame 并导出为 CSV 文件
        df = pd.DataFrame({
            'y_order': y_order.numpy() if isinstance(y_order, torch.Tensor) else y_order,
            'y_pred_db_real': y_pred_db_real.numpy() if isinstance(y_pred_db_real, torch.Tensor) else y_pred_db_real,
            'y_pred_db_fake': y_pred_db_fake.numpy() if isinstance(y_pred_db_fake, torch.Tensor) else y_pred_db_fake,
            'y_true': y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
        })
        # 保存为 CSV 文件
        df.to_csv('eval_results.csv', index=False)
        y_pred_db = np.float32((y_pred_db_real < y_pred_db_fake))
        cnn_accy=self._evaluate(y_order, y_pred_db, y_true)
        _save_dir = os.path.join(self.log, "task" + str(self.stage))
        os.makedirs(_save_dir, exist_ok=True)
        _pred_path = os.path.join(_save_dir, "closed_set_pred.npy")
        _target_path = os.path.join(_save_dir, "closed_set_target.npy")
        np.save(_pred_path, y_pred_db)
        np.save(_target_path, y_true)
        _confusion_img_path = os.path.join(_save_dir, "closed_set_conf.png")
        # plot_confusion(_confusion_img_path, confusion_matrix(y_true, y_pred))
        logging.info("closed-set No NME accuracy.")
        logging.info("closed-set CNN: {}".format(cnn_accy["grouped"]))

    def _evaluate(self, y_order,y_pred_bb, y_true):
        ret = {}
        grouped = accuracy(y_order,y_pred_bb,y_true,self.stage, self.stage)
        ret["grouped"] = grouped
        return ret
    def eval_cnn(self,eval_loader):
        y_pred_db_real,y_pred_db_fake,y_pred_bb, y_true, y_order = [], [], [],[],[]
        self.net.eval()
        with torch.no_grad():
            for i, (_, inputs, targets, orders) in enumerate(eval_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                #b n  b n
                logits_db_real,logits_db_fake=self.get_result(inputs,targets,orders)
                # logits_bb = self.net(inputs)["logits"]
                y_order.append(orders.cpu().numpy())
                y_pred_db_real.append(logits_db_real.detach().cpu().numpy())
                y_pred_db_fake.append(logits_db_fake.detach().cpu().numpy())
                # y_pred_bb.append(logits_bb.cpu().numpy())
                y_true.append(targets.cpu().numpy())
            return np.concatenate(y_order), np.concatenate(y_pred_db_real),np.concatenate(y_pred_db_fake), np.concatenate(y_true)

    def get_result(self,inputs,targets,order):
        """
        Args:
            inputs(tensor):b,c,299,299
            targets(tensor):b
            order(tensor):b
            attn_layer(tensor):b,c,h,w
        """
        layers=self.net(inputs)
        attn_layer=layers[self.attention_layer]
        fea_layer=layers[self.feature_layer]
        imp_attns,imp_locat=self.attentions(attn_layer)
        # B,C,N,window_x,window_y
        extracted_weight,extracted=self.shallow(fea_layer,imp_attns,imp_locat)
        unique_value = int(torch.unique(order)[0].item())
        #logits
        result_sum_real=self.db_retrieve(extracted,self.db_real[unique_value],self.weight_dim[unique_value])
        result_sum_fake=self.db_retrieve(extracted,self.db_fake[unique_value],self.weight_dim[unique_value])
        return result_sum_real,result_sum_fake

def accuracy(y_order, y_pred, y_true, nb_old, increment=1):
    all_acc = {}
    all_acc["total"] = np.around(
        np.sum(y_pred == y_true) * 100 / len(y_true), decimals=2
    )
    # Grouped accuracy
    for class_id in range(0, np.max(y_order) + 1, increment):
        label = "{}".format(
            str(class_id).rjust(2, "0")
        )
        idxes = np.where(y_order == class_id)
        if len(idxes[0]) == 0:
            all_acc[label] = 'nan'
            continue
        all_acc[label] = np.around(
            np.sum(y_pred[idxes] == y_true[idxes]) * 100 / len(idxes[0]), decimals=2
        )
        # real = np.size(np.where((y_pred[idxes] == y_true[idxes]) & (y_true[idxes] == np.zeros_like(y_true[idxes]))))
        # real_total = np.sum(y_true[idxes] == np.zeros_like(y_true[idxes]))
        # fake = np.size(np.where((y_pred[idxes] == y_true[idxes]) & (y_true[idxes] == np.ones_like(y_true[idxes]))))
        # fake_total = np.sum(y_true[idxes] == np.ones_like(y_true[idxes]))
        # logging.info("label: {}, real: {}/{}, fake: {}/{}".format(label, real, real_total, fake, fake_total))

    # Old accuracy
    idxes = np.where(y_order < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            np.sum(y_pred[idxes] == y_true[idxes]) * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_order >= nb_old)[0]
    all_acc["new"] = np.around(
        np.sum(y_pred[idxes] == y_true[idxes]) * 100 / len(idxes), decimals=2
    )

    return all_acc


