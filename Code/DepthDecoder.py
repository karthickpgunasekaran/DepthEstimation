
#LOAD ALL PYTORCH MODULES
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_1(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=1),
        nn.ELU(inplace=True)
    )

def conv_2(in_planes):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, 1, kernel_size=3, stride=1),
        nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes, kernel_size=3,stride=1,padding=0),
        nn.ELU(inplace=True)

    )

def match_size(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class DepthDecoder(nn.Module):
    def __init__(self, alpha=10, beta=0.01):
        super(DepthDecoder, self).__init__()
        self.up_scale = nn.Upsample(scale_factor=2)

        self.alpha = alpha
        self.beta = beta
        conv_dims = [64, 64, 128, 256, 512]
        up_dims =  [512,256, 128, 64, 32, 16]
        self.upconv5 = conv_1(up_dims[0], up_dims[1])
        self.iconv5 = conv(up_dims[1] + conv_dims[3], up_dims[1])

        self.upconv4 = conv_1(up_dims[1], up_dims[2])
        self.iconv4 = conv(up_dims[2] + conv_dims[2], up_dims[2])
        self.disp4 = conv_2(up_dims[2])


        self.upconv3 = conv_1(up_dims[2], up_dims[3])
        self.upconv2 = conv_1(up_dims[3], up_dims[4])
        self.upconv1 = conv_1(up_dims[4], up_dims[5])



        self.iconv3 = conv(up_dims[3] + conv_dims[1], up_dims[3])
        self.iconv2 = conv( up_dims[4] + conv_dims[0], up_dims[4])
        self.iconv1 = conv( up_dims[5], up_dims[5])


        self.disp3 = conv_2(up_dims[3])
        self.disp2 = conv_2(up_dims[4])
        self.disp1 = conv_2(up_dims[5])

    def forward(self,x,econv):
        # Upconvolution
        DEBUG_DEC = 0
        #Part 1
        pred_upconv5 = match_size(F.interpolate(self.upconv5(econv[4]), scale_factor=2, mode='bilinear', align_corners=False), econv[3])
        concat5 = torch.cat((pred_upconv5, econv[3]), 1)
        pred_iconv5 = self.iconv5(concat5)

        #Part 2
        pred_upconv4 = match_size(F.interpolate(self.upconv4(pred_iconv5), scale_factor=2, mode='bilinear', align_corners=False), econv[2])
        concat4 = torch.cat((pred_upconv4, econv[2]), 1)
        pred_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.disp4(pred_iconv4) + self.beta

        if DEBUG_DEC == 1:
            print("DISP 4:",disp4.shape)

        #Part 3
        out_upconv3 = match_size(F.interpolate(self.upconv3(pred_iconv4), scale_factor=2, mode='bilinear', align_corners=False), econv[1])
        concat3 = torch.cat((out_upconv3, econv[1]), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.disp3(out_iconv3) + self.beta
        if DEBUG_DEC == 1:
            print("DISP 3:",disp3.shape)

        #Part 4
        out_upconv2 = match_size( F.interpolate(self.upconv2(out_iconv3), scale_factor=2, mode='bilinear', align_corners=False),econv[0])
        concat2 = torch.cat((out_upconv2, econv[0]), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.disp2(out_iconv2) + self.beta

        if DEBUG_DEC == 1:
            print("DISP 2:",disp2.shape)

        out_upconv1 = match_size( F.interpolate(self.upconv1(out_iconv2), scale_factor=2, mode='bilinear', align_corners=False),x)
        out_iconv1 = self.iconv1(out_upconv1)
        disp1 = self.alpha * self.disp1(out_iconv1) + self.beta

        if DEBUG_DEC == 1:
            print("DISP 1:",disp1.shape)

        if self.training == True:
            return [disp1,disp2,disp3,disp4]
        else:
            return disp1
