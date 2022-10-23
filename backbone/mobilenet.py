import torch
import torch.nn as nn
import torchvision

# Mobilenet paper: https://arxiv.org/pdf/1905.02244.pdf

class Mobilenet_large(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()

        if pretrained == True:
            self.model = torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V2')
        else:
            self.model = torchvision.models.mobilenet_v3_large(weights= None)


        self.depth_channels = [16, 24, 40, 112, 960]

        del self.model.avgpool
        del self.model.classifier

    def forward(self, x):
        p1 = self.model.features[:2] #feature map shape: Batch_size x 16 x input_H / 2 x input_W / 2
        p2 = self.model.features[2:4] #feature map shape: Batch_size x 24 x input_H / 4 x input_W / 4
        p3 = self.model.features[4:7] #feature map shape: Batch_size x 40 x input_H / 8 x input_W / 8
        p4 = self.model.features[7:13] #feature map shape: Batch_size x 112 x input_H / 16 x input_W / 16 
        p5 = self.model.features[13:] #feature map shape: Batch_size x 960 x input_H / 32 x input_W / 32
        

        return p1, p2, p3, p4, p5


class Mobilenet_small(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()
        if pretrained == True:
            self.model = torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V2')
        else:
            self.model = torchvision.models.mobilenet_v3_large(weights= None)
        
        self.depth_channels = [16, 16, 24, 48, 576]

        #del self.model.avgpool
        #del self.model.fc

    def forward(self, x):
        p1 = self.model.features[0] #feature map shape: Batch_size x 16 x input_H / 2 x input_W / 2
        p2 = self.model.features[1] #feature map shape: Batch_size x 16 x input_H / 4 x input_W / 4       
        p3 = self.model.features[2:4] #feature map shape: Batch_size x 24 x input_H / 8 x input_W / 8      
        p4 = self.model.features[4:9] #feature map shape: Batch_size x 48 x input_H / 16 x input_W / 16      
        p5 = self.model.features[9:] #feature map shape: Batch_size x 576 x input_H / 32 x input_W / 32
        

        return p1, p2, p3, p4, p5
