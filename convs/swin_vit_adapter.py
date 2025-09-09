import math

import timm.models.swin_transformer
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        # self.adapter_down = nn.Conv2d(in_dim, out_dim, 1, 1, 1)
        # self.adapter_up = nn.Conv2d(in_dim, out_dim, 1, 1, 1)
        self.adapter_down = nn.Linear(in_dim, out_dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(out_dim, in_dim)  # equivalent to 1 * 1 Conv

        # self.adapter_gate = Gate()
        self.relu = nn.ReLU(inplace=True)
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
        x = self.relu(x_down)
        x_up = self.adapter_up(x)
        return x_up


def forward_block(self, x):
    B, H, W, C = x.shape
    x = x + self.drop_path1(self._attn(self.norm1(x))) + 1 * self.drop_path1(self.adapter(x.reshape(B, -1, C))).reshape(B, H, W, C)
    x = x.reshape(B, -1, C)
    x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x = x.reshape(B, H, W, C)
    return x


def set_adapter(model):
    for _ in model.children():
        if type(_) == timm.models.swin_transformer.SwinTransformerBlock:
            _.adapter = Adapter(_.dim)
            bound_method = forward_block.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_adapter(_)
