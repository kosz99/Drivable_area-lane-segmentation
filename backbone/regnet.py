import torch
import torchvision
import torch.nn as nn

# Regnet paper: https://arxiv.org/pdf/2101.00590.pdf


class RegNet(nn.Module):
    def __init__(self, pretrained = True):
        super().__init__()
        if pretrained == True:
            self.model = torchvision.models.regnet_y_1_6gf(weights='IMAGENET1K_V2')
        else:
            self.model = torchvision.models.regnet_y_1_6gf(weights= None)
        
        
        self.depth_channels = [32, 48, 120, 336, 888]
        
        del self.model.avgpool
        del self.model.fc
    
    def forward(self, X):
        p1 = self.model.stem(X)     #feature map shape: Batch_size x 32 x input_H / 2 x input_W / 2
        p2 = self.model.trunk_output[:1](p1)    #feature map shape: Batch_size x 48 x input_H / 4 x input_W / 4
        p3 = self.model.trunk_output[1:2](p2)   #feature map shape: Batch_size x 120 x input_H / 8 x input_W / 8
        p4 = self.model.trunk_output[2:3](p3)   #feature map shape: Batch_size x 336 x input_H / 16 x input_W / 16
        p5 = self.model.trunk_output[3:4](p4)   #feature map shape: Batch_size x 888 x input_H / 32 x input_W / 32
        
        return p1, p2, p3, p4, p5
