import torch
import torch.nn as nn
# FPN implemented according to: https://arxiv.org/pdf/1612.03144.pdf


class FPN(nn.Module):
    def __init__(self, depth_channels, d):
        super().__init__()
        self.depth = len(depth_channels)
        # conv1x1 - adjust backbone feature maps depth to fpn depth
        self.convs1x1 = nn.ModuleList([nn.Conv2d(depth_channels[i], d, 1, bias=False) for i in range(self.depth)])
        #conv3x3 - anti aliasing
        self.convs3x3 = nn.ModuleList([nn.Conv2d(d, d, 3, 1, 1, bias=False) for i in range(self.depth-1)])

    def forward(self, feature_maps):
        assert len(feature_maps) == self.depth
        results = []
        res = self.convs1x1[-1](feature_maps[-1])
        results.append(res)

        for i in range(self.depth-1):
            res = nn.Upsample((feature_maps[-2-i].shape[2], feature_maps[-2-i].shape[3]), mode='nearest')(res)
            add = self.convs1x1[-2-i](feature_maps[-2-i])
            res = res+add
            res = self.convs3x3[i](res)
            results.append(res)
        
        return results

class FPN_only_last_map(nn.Module):
    def __init__(self, depth_channels, d):
        super().__init__()
        self.depth = len(depth_channels)
        # conv1x1 - adjust backbone feature maps depth to fpn depth
        self.convs1x1 = nn.ModuleList([nn.Conv2d(depth_channels[i], d, 1, bias=False) for i in range(self.depth)])
        #conv3x3 - anti aliasing
        self.convs3x3 = nn.ModuleList([nn.Conv2d(d, d, 3, 1, 1, bias=False) for i in range(self.depth-1)])

    def forward(self, feature_maps):
        assert len(feature_maps) == self.depth
        res = self.convs1x1[-1](feature_maps[-1])
    
        for i in range(self.depth-1):
            res = nn.Upsample((feature_maps[-2-i].shape[2], feature_maps[-2-i].shape[3]), mode='nearest')(res)
            add = self.convs1x1[-2-i](feature_maps[-2-i])
            res = res+add
            res = self.convs3x3[i](res)

        return res
