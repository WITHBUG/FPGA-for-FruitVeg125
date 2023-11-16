"""
This network is quantized and used for emulation inference in FPGA.
"""

import torch
import torch.nn as nn


def myround(ten):
    '''if tensor%1 == 0.5, then +1 '''
    
    # out, in_c, hei, wei = ten.shape

    # test = torch.zeros(out, in_c, hei, wei)
    # test = test + 0.5

    # bo = test.eq(torch.frac(ten))
    # for o in range(out):
    #     for i in range(in_c):
    #         for h in range(hei):
    #             for w in range(wei):
    #                 if bo[o, i, h, w]:
    #                     ten[o, i, h, w] = torch.floor(ten[o, i, h, w]) + 1

    # ten = torch.round(ten.clone().detach())
    # return ten
    is_05 = torch.frac(ten) == 0.5
    Ten = torch.where(is_05, torch.ceil(ten), torch.round(ten))
    return Ten.detach()


# The channels needs to be set manually
out_channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]

# Custom convolution layer for simulating quantization calculations.
class QuantizedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        # those two parameters are used for quantization of activation output
        self.register_buffer('constant', torch.tensor(1))
        self.register_buffer('divisor', torch.tensor(2 ** 14))
        
    def forward(self, input):
        # self._conv_forward's function is to call nn.functional.conv2d()
        out = self._conv_forward(input, self.weight, self.bias)
        # Note! Here the function can be torch.round or torch.trunc or torch.floor
        # print(self.constant)
        # print(self.divisor)
        # out = torch.round(out * self.constant / self.divisor + torch.tensor(0.000001))
        out = myround(out * self.constant / self.divisor)
        print(out.shape)
        # execute ReLU and clamp the values over 255 
        return torch.clamp(out, 0, 255)

class DWConv(nn.Module):
    # no bn and relu layer
    def __init__(self, in_channels, out_channels, stride):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(
            # dw
            QuantizedConv(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=True),
            # pw
            QuantizedConv(in_channels, out_channels, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        return self.conv(x)

class QuanNet(nn.Module):
    def __init__(self, n_class = 125):
        super().__init__()
        
        # Start Conv
        self.conv1 = nn.Sequential(
            QuantizedConv(3, out_channels[0], 3, 2, 1, bias=True),

        )
        # DWConv blocks
        self.conv2 = DWConv(out_channels[0], out_channels[1], 1)
        self.conv3 = DWConv(out_channels[1], out_channels[2], 2)
        self.conv4 = DWConv(out_channels[2], out_channels[3], 1)
        self.conv5 = DWConv(out_channels[3], out_channels[4], 2)
        self.conv6 = DWConv(out_channels[4], out_channels[5], 1)
        self.conv7 = DWConv(out_channels[5], out_channels[6], 2)
        self.conv8 = DWConv(out_channels[6], out_channels[7], 1)
        self.conv9 = DWConv(out_channels[7], out_channels[8], 1)
        self.conv10 = DWConv(out_channels[8], out_channels[9], 1)
        self.conv11 = DWConv(out_channels[9], out_channels[10], 1)
        self.conv12 = DWConv(out_channels[10], out_channels[11], 1)
        self.conv13 = DWConv(out_channels[11], out_channels[12], 2)
        self.conv14 = DWConv(out_channels[12], out_channels[13], 1)
  
        # Classifier
        self.classifier = nn.Linear(out_channels[-1], n_class)

    def forward(self, x):  
        # feature extraction
        x = self.conv1(x)                            
        x = self.conv2(x)                             
        x = self.conv3(x)                         
        x = self.conv4(x)                               
        x = self.conv5(x)                               
        x = self.conv6(x)                     
        x = self.conv7(x)
        x = self.conv8(x)                                   # out: B * 512 * 14 * 14
        x = self.conv9(x)                                   # out: B * 512 * 14 * 14
        x = self.conv10(x)                                  # out: B * 512 * 14 * 14
        x = self.conv11(x)                                  # out: B * 512 * 14 * 14
        x = self.conv12(x)                                  # out: B * 512 * 14 * 14
        x = self.conv13(x)                                  # out: B * 1024 * 7 * 7
        x = self.conv14(x)                                  # out: B * 1024 * 7 * 7                
 
        # classifier
        # Note! only addition is done here, not averaging
        x = x.sum(3).sum(2)                    
        x = self.classifier(x)                 
        
        return x
    

if __name__ == "__main__":
    net = QuanNet()
    a = torch.randint(0, 255, size=(1, 3, 224, 224)).float()
    b = net(a)
    print(b)
