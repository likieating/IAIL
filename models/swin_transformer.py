import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

        Args:
            input_resolution (tuple[int]): Resolution of input feature.
            dim (int): Number of input channels.
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution=input_resolution
        self.dim=dim
        self.reduction=nn.Linear(4*dim,2*dim,bias=False)
        self.norm=norm_layer(2*dim)

    def forward(self,x):
        """
            x: B, H*W, C
        """
        H,W=self.input_resolution
        B,L,C=x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0=x[:,0::2,0::2,:]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

def window_reverse(windows,window_size,H,W)
    """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
    B=int(windows.shape[0]/(H*W/window_size/window_size))
    x=windows.view(B,H//window_size,W//window_size,window_size,window_size,-1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        super().__init__()
        out_features=out_features or in_features
        hidden_features=hidden_features or in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout(drop)

    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x = self.drop(x)
        x=self.fc2(x)
        x = self.drop(x)
        return x



def window_partition(x,window_size):
    """
        x:b,h,w,c
        window_size(int):window size
    """
    B,H,W,C=x.shape
    x=x.view(B,H//window_size,window_size,W//window_size,window_size,C)
    windows=x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
    # b*h*w/window/window window window embedding
    # num_windows window_size window_size
    return windows


class SwinTransformerBlock(nn.Module):
     """
     Args:
         dim:number of input channels
         input_resolution
         window_size:Window size
         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
         shift_size:shift window size
         drop (float, optional): Dropout rate. Default: 0.0
         norm_layer:normalization layer
         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
         drop_path:Stochastic depth rate. Default: 0.0
     """
     def __init__(self,dim,input_resolution,window_size=7,drop_path=0.,
                  norm_layer=nn.LayerNorm,mlp_ratio=4.,act_layer=nn.GELU,
                  drop=0.,shift_size=0,num_heads=3,qkv_bias=True,attn_drop=0.,
                  pretrained_window_size=0):
         super().__init__()
         self.dim=dim
         self.input_resolution=input_resolution
         self.window_size=window_size
         self.shift_size=shift_size
         self.mlp_ratio = mlp_ratio
         if min(self.input_resolution)<= self.window_size:
             self.shift_size=0
             self.window_size=min(self.input_resolution)
         assert 0<=self.shift_size<self.window_size,"shift_size must in 0-window_size"

         self.norm1=norm_layer(dim)
         self.attn=WindowAttention(
             dim,window_size=to_2tuple(self.window_size),num_heads=num_heads,
             qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=drop,
             pretrained_window_size=to_2tuple(pretrained_window_size)
         )

         self.drop_path=DropPath(drop_path) if drop_path>0. else nn.Identity()
         self.norm2=norm_layer(dim)
         mlp_hidden_dim=int(dim*mlp_ratio)
         self.mlp=Mlp(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)

         if self.shift_size>0:
             H,W=self.input_resolution
             img_mask=torch.zeros((1,H,W,1))
             h_slices=(slice(0,-self.window_size),
                       (-self.window_size,-self.shift_size),
                       slice(-self.shift_size,None))
             w_slices = (slice(0, -self.window_size),
                         slice(-self.window_size, -self.shift_size),
                         slice(-self.shift_size, None))
             cnt=0
             for h in h_slices:
                 for w in w_slices:
                     img_mask[:,h,w,:]=cnt
                     cnt+=1
             # nW, window_size, window_size, 1
             mask_windows=window_partition(img_mask,self.window_size)
             mask_windows=mask_windows.view(-1,self.window_size*self.window_size)
             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
             attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
         else:
             attn_mask = None

         self.register_buffer("attn_mask",attn_mask)





     # batch,features,embedding
     def forward(self,x):
         H,W=self.input_resolution
         B,L,C=x.shape
         assert L == H * W, "input feature has wrong size"
         shortcut=x
         x=x.view(B,H,W,C)

         #cyclic shift
         if self.shift_size > 0:
             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
         else:
             shifted_x = x

         # num_windows=nW*B window_size window_size C
         x_windows=window_partition(shifted_x,self.window_size)
         # num_windows each_window C
         x_windows=x_windows.view(-1,self.window_size*self.window_size,C)

         #W-MSA/SW-MSA
         attn_windows=self.attn(x_windows,mask=self.attn_mask)

         #merge windows
         attn_windows=attn_windows.view(-1,self.window_size,self.window_size,C)
         # B H' W' C
         shifted_x=window_reverse(attn_windows,self.window_size,H,W)

         #reverse cyclic shift
         if self.shift_size > 0:
             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
         else:
             x = shifted_x
         x=x.view(B,H*W,C)
         x=shortcut+self.drop_path(self.norm1(x))

         # FFN
         x = x + self.drop_path(self.norm2(self.mlp(x)))

         return x











class WindowAttention(nn.Module):
    """
    Args:
        dim:input dimension
        window_size:The height and width of the window.
        num_heads=Number of attention heads
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
        qkv_bias:If True, add a learnable bias to query, key, value. Default: True
    """
    def __init__(self,dim,window_size,num_heads,
                 pretrained_window_size=[0, 0],qkv_bias=True,
                 attn_drop=0.):
        super().__init__()
        self.dim=dim
        self.window_size=window_size
        self.pretrained_window_size=pretrained_window_size
        self.num_heads=num_heads
        #optimize the heads
        self.logit_scale=nn.Parameter(torch.log(10*torch.ones((num_heads,1,1))),requires_grad=True)

        #mlp to gernerate continuos relative position bias
        self.cpb_mlp=nn.Sequential(nn.Linear(2,512,bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(512,num_heads,bias=True))

        #get relative_coords_table
        relative_coord_h=torch.arange(-(self.window_size[0]-1),self.window_size[0])
        relative_coord_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1])
        relative_coord_table=torch.stack(
            torch.meshgrid([relative_coord_h,relative_coord_w])).permute(1,2,0).contiguous().unsqueeze(0) #(1,2*wh-1,2*ww-1,2)
        if pretrained_window_size[0]>0:
            relative_coord_table[:,:,:,0]/=(pretrained_window_size[0])
            relative_coord_table[:, :, :, 1] /= (pretrained_window_size[1])
        else:
            relative_coord_table[:, :, :, 0] /= (self.window_size[0])
            relative_coord_table[:, :, :, 1] /= (self.window_size[1])
        relative_coord_table*=8 #normalize to (-8,8)
        #log normalize
        relative_coord_table=torch.sign(relative_coord_table)*torch.log2(abs(
            relative_coord_table)+1)/np.log2(8)
        self.register_buffer("relative_coords_table",relative_coord_table)

        #get pair-wise relative position index for each token inside the window
        coords_h=torch.arange(self.window_size[0])
        coords_w=torch.arange(self.window_size[1])
        coords=torch.stack(torch.meshgrid([coords_h,coords_w]))
        coords_flatten=torch.flatten(coords,1)# 2 wh*wv
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv=nn.Linear(dim,dim*3,bias=False)
        if qkv_bias:
            self.q_bias=nn.Parameter(torch.zeros(dim))
            self.v_bias=nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias=None
            self.v_bias =None
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop(nn.Dropout)
        self.softmax=nn.Softmax(dim=1)



    def forward(self,x,mask=None):
        #batch tokens features
        #num_windows tokens C
        B_,N,C=x.shape
        qkv_bias=None
        if self.q_bias is not None:
            #3*c
            qkv_bias=torch.cat((self.q_bias,torch.zeros_like(self.v_bias,requires_grad=False),self.v_bias))
        #num_windows tokens C*3
        qkv=F.linear(input=x,weight=self.qkv.weight,bias=qkv_bias)
        #num_windows each_window 3 num_heads embedding/num_heads
        #3 num_windows num_heads each_window embedding/num_heads
        qkv=qkv.reshape(B_,N,3,self.num_heads,-1).permute(2,0,3,1,4)

        q,k,v=qkv[0],qkv[1],qkv[2]


        #cosine attention
        attn =(F.normalize(q,dim=-1) @ F.normalize(k,dim=-1).transpose(-2,-1))
        logit_scale=torch.clamp(self.logit_scale,max=torch.log(torch.tensor(1./0.01))).exp()
        attn=attn*logit_scale

        relative_position_bias_table=self.cpb_mlp(self.relative_coords_table).view(-1,self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn=attn+relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW=mask.shape[0]
            attn=attn.view(B_//nW,nW,self.num_heads,N,N)+mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn=self.attn_drop(attn)

        x=(attn@v).transpose(1,2).reshape(B_,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
















class BasicLayer(nn.Module):
    """
    Args:
        dim(int):Number of input channels
        input_resolution:Input resolution
        depth:number of blocks
        window_size:Local window size
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.

    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.dim=dim
        self.input_resolution=input_resolution
        self.depth=depth
        self.use_checkpoint = use_checkpoint

        self.blocks=nn.ModuleList(
            [
                SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer,
                                     pretrained_window_size=pretrained_window_size)
                for i in range(depth)
            ]
        )
        if downsample is not None:
            self.downsample =downsample(input_resolution,dim=dim,norm_layer=norm_layer)
        else:
            self.downsample=None


    def forward(self,x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x=checkpoint.checkpoint(blk,x)
            else:
                x=blk(x)

        if self.downsample is not None:
            x=self.downsample(x)
        return x





class swinT(nn.Module):
    """
    args:
        img_size(int):input img size
        patch_size(int):4
        in_chan(int):input imgs' channels
        num_classes(int):output classes
        embed_dim(int):patch embedding dimension. Default:96
        depths:num of layers in each stage
        number_heads:
        window_size:
        mlp_ratio:
        qkv_bias(float):if true,add a learnabel bias to query
        drop_rate(float):Dropout rate,Default：0
        attn_drop_rate(float):
        drop_path_rate(float):Stochastic depth rate.Default: 0.1
        norm_layer(nn.Module):Normalization layer,Default:nn.LayerNorm
        ape(bool):absolute position embedding,Default:false
        patch_norm
        use_checkpoint(bool):Whether to use checkpoint to save memory,Default:False
        pretrained_window_size:
    """



    def __init__(self,img_size=224,patch_size=4,in_chan=3,num_classes=2,
                 embed_dim=96,depths=[2,2,6,2],norm_layer=nn.LayerNorm,ape=False,
                 drop_rate=0.,drop_path_rate=0.1):
        super().__init__()
        self.num_classes=num_classes
        self.norm_layer=norm_layer
        self.input_resolution=inp
        self.num_layers=len(depths)
        # split the imgs into different patches
        self.patch_embed=PatchEmbed(img_size,patch_size,in_chan,embed_dim,norm_layer if self.norm_layer else None)
        patch_resolution=self.patch_embed.patch_resolution
        num_patches=self.patch_embed.num_patches

        #absolute position embedding
        if self.ape:
            self.absolute_pos_embed=nn.Parameter(torch.zeros(1,num_patches,embed_dim))
            trunc_normal_(self.absolute_pos_embed,std=.02)

        self.pos_drop=nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr=[x.item() for x in torch.linspace(0,drop_path_rate,sum(depths))]

        #build layers(each layer may have repeated blocks)
        self.layers=nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer=BasicLayer(
                dim=int(embed_dim*2**i_layer)
            )



    def forward_features(self,x):
        x=self.patch_embed(x)
        if self.ape:
            x=x+self.absolute_pos_embed
        x.pos_drop(x)

        for layer in self.layers:
            x=layer(x)


    def forward(self,x):
        x=self.forward_features(x)







class PatchEmbed(nn.Module):
    """
    Args:
        img_size=224
        patch_size:patch token size
        in_chans
        embed_dim:output channels
        norm_layer:normalization layer
    """
    def __init__(self,img_size=224,patch_size=4,in_chans=3,embed_dim=96,norm_layer=None):
        super().__init__()
        img_size=to_2tuple(img_size)
        patch_size=to_2tuple(patch_size)
        patch_resolution=[img_size[0]//patch_size[0],img_size[0]//patch_size[0]]
        self.img_size=img_size
        self.patch_size=patch_size
        self.patch_resolution=patch_resolution
        self.num_patches=patch_resolution[0]*patch_resolution[1]

        self._in_chans=in_chans
        self.embed_dim=embed_dim
        self.proj=nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size)
        if norm_layer is not None:
            self.norm=norm_layer(embed_dim)
        else:
            self.norm_layer=None

        def forward(self,x):
            """
            x:input imgs(batch,channel,height,width)
            """
            B,C,H,W=x.shape
            assert H==self.img_size[0] and W==self.img_size[1],\
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            # (B,embed_dim,H,W)->(B,H*W,embed)
            x=self.proj(x).flatten(2).transpose(1,2)
            if self.norm is not None:
                x=self.norm(x)
            return x

