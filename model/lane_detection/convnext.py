# Convnext model
import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.normpath( os.path.join(os.getcwd(), *([".."] * 1))))

from backbone.convnext import ConvNext
from neck.fpn import FPN_only_last_map
from head.head import simple_conv_segmentation_head




class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.backbone = ConvNext()
        self.neck = FPN_only_last_map(self.backbone.depth_channels, 64)
        self.output_lane = simple_conv_segmentation_head(64, "LeakyReLU", output_depth=1)
        

    def forward(self, x):
        features = self.backbone(x)
        neck_features = self.neck(features)


        return self.sigmoid(self.output_lane(neck_features))





