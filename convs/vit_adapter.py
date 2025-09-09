import math

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import VisionTransformer, Block, \
    checkpoint_filter_fn
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, \
    checkpoint_seq
import torch.nn as nn
import torch
from functools import partial


class Adapter(nn.Module):
    def __init__(self, dim=8):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv

        # self.adapter_gate = Gate()
        self.gelu = nn.GELU()
        self.dim = dim
        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x = self.gelu(x_down)
        x_up = self.adapter_up(x)
        return x_up


class BlockWithAdapter(Block):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qkv_norm=False, proj_drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=Mlp, s=0.1):
        super(BlockWithAdapter, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qkv_norm, proj_drop, attn_drop,
                                               init_values, drop_path, act_layer, norm_layer, mlp_layer)
        # self.adapter_attn = Adapter(64)
        self.adapter_mlp = Adapter(64)
        self.s = s

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x2 = self.adapter_mlp(self.norm2(x))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))) + self.drop_path2(self.ls2(x2)) * self.s
        return x


class ViTAdapter(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
                 class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.,
                 weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block,
                 *args, **kwargs):
        # Below are copied from as VisionTransformer
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False


        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            BlockWithAdapter(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                              init_values=init_values, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                              act_layer=act_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.iteration = 0

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return {'features': x}

    def forward(self, x):
        out = self.forward_features(x)
        x = out['features']
        x = self.forward_head(x)
        return x, out


def _create_vit_adapter(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        ViTAdapter, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn, **kwargs)
    return model


def vit_adapter_patch16_224(pretrained: bool = False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vit_adapter('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
