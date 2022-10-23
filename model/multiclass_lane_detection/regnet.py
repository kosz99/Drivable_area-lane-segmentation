# Convnext model
import torch
import torch.nn as nn

import os
import sys
#sys.path.append(os.getcwd())
sys.path.append(os.path.normpath( os.path.join(os.getcwd(), *([".."] * 1))))

from backbone.regnet import RegNet
from neck.fpn import FPN
from head.head import segmentation_head




class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = RegNet()
        self.neck = FPN(self.backbone.depth_channels[1:], 64)
        self.norm = nn.BatchNorm2d(64)
        self.output_lane = segmentation_head(64,64, "LeakyReLU", 10)
    
    
    
        
    def forward(self, x):
        features = self.backbone(x)
        neck_features = self.neck(features[1:])


        return self.output_lane(self.norm(neck_features[-1]))


