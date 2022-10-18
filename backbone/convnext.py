import torch
import torch.nn as nn
import torchvision

# Convnext paper: https://arxiv.org/pdf/2201.03545.pdf

class ConvNext(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()

        if pretrained == True:
            self.model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        else:
            self.model = torchvision.models.convnext_tiny(weights=None)
        
        self.depth_channels = [96, 192, 384, 768]

        del self.model.avgpool
        del self.model.fc
    
    def forward(self, x):
        p1 = self.model.features[:2](x) #feature map shape: Batch_size x 96 x input_H / 4 x input_W / 4
        p2 = self.model.features[2:4](p1) #feature map shape: Batch_size x 192 x input_H / 8 x input_W / 8
        p3 = self.model.features[4:6](p2) #feature map shape: Batch_size x 384 x input_H / 16 x input_W / 16
        p4 = self.model.features[6:](p3) #feature map shape: Batch_size x 768 x input_H / 32 x input_W / 32
        
        return p1, p2, p3, p4
