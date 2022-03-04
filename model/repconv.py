import torch.nn as nn

import torch
import torch.nn.functional as F

EXPAND_RATIO = 4
class repconv(nn.Module):
    def __init__(
        self, channels, out_channels,kernel_size,
            stride=1, dilation=1, groups=1):
        super(repconv, self).__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        padding = int((kernel_size - 1) / 2) * dilation
        self.padding = padding
        self.shortcut = self.channels==self.out_channels
        self.update = False
        # convlution kernel for testing
        self.conv = nn.Conv2d(channels, out_channels, kernel_size, stride, padding=padding, bias=True,
                               dilation=dilation, groups=groups)
        # convlution kernel for training
        self.conv_1x1 = nn.Conv2d(channels, out_channels, 1, stride, padding=0, bias=True,
                               dilation=dilation, groups=groups)
        if self.kernel_size == 3:
            self.conv_1x3 = nn.Conv2d(channels, out_channels, (1, kernel_size), stride, padding=(0,padding), bias=True,
                                   dilation=dilation, groups=groups)
            self.conv_3x1 = nn.Conv2d(channels, out_channels, (kernel_size,1), stride, padding=(padding, 0), bias=True,
                                   dilation=dilation, groups=groups)
            self.conv_3x3 = nn.Conv2d(channels, out_channels, kernel_size, stride, padding=padding, bias=True,
                                   dilation=dilation, groups=groups)
        self.conv_expand = nn.Conv2d(out_channels, int(EXPAND_RATIO * out_channels), 1, 1,0,bias=True)
        self.conv_shrink = nn.Conv2d(int(EXPAND_RATIO * out_channels), out_channels, 1, 1, 0, bias=True)


    def rep(self):
        self.conv =  nn.Conv2d(self.channels, self.out_channels, self.kernel_size, self.stride, padding=self.padding, bias=True,
                                   dilation=self.dilation, groups=self.groups)
        # base weight
        if self.kernel_size == 3:
            weight, bias = self.conv_3x3.weight, self.conv_3x3.bias # out_c*in_c*3*3, out_c
            # pad 1x3, 3x1 and 1x1
            weight = weight + F.pad(self.conv_1x3.weight,[0,0,1,1]) + \
                     F.pad(self.conv_3x1.weight,[1,1,0,0]) + F.pad(self.conv_1x1.weight,[1,1,1,1])
            bias = bias + self.conv_1x3.bias + self.conv_3x1.bias + self.conv_1x1.bias
        else:
            weight, bias = self.conv_1x1.weight, self.conv_1x1.bias  # out_c*in_c*1*1, out_c
        #expand
        weight = F.conv2d(weight.permute(1, 0, 2, 3), self.conv_expand.weight, padding=0).permute(1, 0, 2, 3)
        bias = (bias.view(1, self.out_channels, 1, 1) * self.conv_expand.weight).sum(
            dim=(1, 2, 3)) + self.conv_expand.bias
        # shrink
        weight = F.conv2d(weight.permute(1, 0, 2, 3), self.conv_shrink.weight, padding=0).permute(1, 0, 2, 3)
        bias = (bias.view(1, EXPAND_RATIO * self.out_channels, 1, 1) * self.conv_shrink.weight).sum(
            dim=(1, 2, 3)) + self.conv_shrink.bias

        if self.shortcut:
            kernel_value = torch.zeros((self.channels, self.channels, self.kernel_size, self.kernel_size))
            for i in range(self.channels):
                kernel_value[i, i, self.kernel_size//2, self.kernel_size//2] = 1
            weight = weight + kernel_value.to(weight.device)
        # assign param
        self.conv.weight = nn.Parameter(weight,requires_grad=False)
        self.conv.bias = nn.Parameter(bias,requires_grad=False)
        self.update = False

        # self.__delattr__('conv_1x3')
        # self.__delattr__('conv_1x1')
        # self.__delattr__('conv_3x1')
        # self.__delattr__('conv_3x3')
        # self.__delattr__('conv_shrink')
        # self.__delattr__('conv_expand')

    def forward(self, x):
        if not self.training:
            if self.update:
                self.rep()
            return self.conv(x)
        else:
            self.update = True
            if self.kernel_size == 3:
                out = self.conv_1x3(x)+self.conv_3x1(x)+self.conv_3x3(x)+self.conv_1x1(x)
            else:
                out = self.conv_1x1(x)
            out = self.conv_shrink(self.conv_expand(out))
            if self.shortcut:
                return x+out
            else:
                return out


if __name__ == '__main__':
    device = torch.device('cuda')
    # build netwark
    c1 = repconv(3,32,3) # 3x3, w/o residual connection
    c2 = repconv(32,32,1) # 1x1, with residual connection
    c3 = repconv(32,32,3) # 3x3, with residual connection
    c4 = repconv(32,3,1) # 1x1, w/o residual connection
    c1.to(device), c2.to(device), c3.to(device), c4.to(device)
    # generate input image
    x = torch.randn(1,3,100,100)
    x = x.to(device)
    # train forward
    y1 = c4(F.relu(c3(F.relu(c2(F.relu(c1(x)))))))
    # reparameter
    c1.eval(), c2.eval(), c3.eval(), c4.eval()
    # test forward
    y2 = c4(F.relu(c3(F.relu(c2(F.relu(c1(x)))))))
    # error
    print((y1-y2).abs().mean()) # 2.8680e-08
