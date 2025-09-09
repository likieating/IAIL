from .base import BaseLearner
from convs.efficientNet import EfficientNet
import torch.nn as nn
import torch.nn.functional as F
import convs.swinTv2 as
depth=4

class djyNet(BaseLearner):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self._old_network = None

        self.layers=nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        swinT(),

                    ]
                )
            )







    def forward(self,x):
        rgb=x
        B=rgb.size(0)
        layers={}
        rgb=EfficientNet.extract_textures(rgb,layers)




