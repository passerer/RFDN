import torch
import torch.nn as nn
import model.block as B

def make_model(args, parent=False):
    model = RFDN()
    return model


class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=50,out_nc=3, upscale=4, deploy = False):
        super(RFDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, 3)

        self.B1 = B.RFDB(nf)
        self.B2 = B.RFDB(nf)
        self.B3 = B.RFDB(nf)
        self.B4 = B.RFDB(nf)
        self.c = B.conv_block(nf * 4, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, 3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx