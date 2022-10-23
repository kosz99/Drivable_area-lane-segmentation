import torch
import torch.nn as nn


def get_activation_function(name: str):
    func = {
        "ReLU": nn.ReLU,
        "ELU": nn.ELU,
        "CELU": nn.CELU,
        "ReLU6": nn.ReLU6,
        "LeakyReLU": nn.LeakyReLU,
        "SELU": nn.SELU

    }
    return func[name]()

class simple_segmentation_head(nn.Module):
    def __init__(self, d: int, act: str = "ReLU"):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.anti_alias_conv1 = nn.Conv2d(d, d, 3, 1, 1)
        self.anti_alias_conv2 = nn.Conv2d(d, 1, 3, 1, 1)
        self.internal_act = get_activation_function(act)
    
    def forward(self, x):
        '''
        Input: Last FPN feature map (shape: H/4, W/4 of output shape)
        Upsampling (x2)
        3x3 anti aliasing conv
        non linear activation function
        upsamplig (x2)
        3x3 anti aliasing conv
        '''
        x = self.upsampling(x)
        x = self.anti_alias_conv1(x)
        x = self.internal_act(x)
        x = self.upsampling(x)
        output = self.anti_alias_conv2(x)

        return output


class simple_conv_segmentation_head(nn.Module):
    def __init__(self, d: int, act: str = "ReLU", output_depth: int = 2):
        super().__init__()

        self.deconv_1 = nn.ConvTranspose2d(d, d, 2, 2, bias=False)
        self.internal_act = get_activation_function(act)
        self.deconv_2 = nn.ConvTranspose2d(d, output_depth, 2, 2, bias=False)

    def forward(self, x):
        '''
        Input: Last FPN feature map (shape: H/4, W/4 of output shape)
        deconvolution - upsampling (x2)
        non linear activation function
        deconvolution - upsampling (x2)
        '''
        x = self.deconv_1(x)
        x = self.internal_act(x)
        output = self.deconv_2(x)

        return output

class segmentation_head(nn.Module):
    def __init__(self, input_depth: int, d: int, act: str = "ReLU", output_depth: int = 1):
        super().__init__()

        self.internal_act = get_activation_function(act)
        
        self.deconv_1 = nn.ConvTranspose2d(input_depth, d, 2, 2, bias=False)
        self.conv_1 = nn.Conv2d(d, d, 3, 1, 1, bias = False)
        
        self.norm = nn.BatchNorm2d(d)
        
        self.deconv_2 = nn.ConvTranspose2d(d, output_depth, 2, 2, bias=False)
        self.conv_2 = nn.Conv2d(output_depth, output_depth, 3, 1, 1, bias = False)

    def forward(self, x):

        x = self.deconv_1(x)
        x = self.conv_1(x)
        x = self.norm(x)
        x = self.internal_act(x)
        x = self.deconv_2(x)
        output = self.conv_2(x)

        return output
