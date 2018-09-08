import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_relu(in_channels, out_channels):
    unit = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
            )
    return unit

def deconv_bn_relu(in_channels, out_channels, drop_out=False):
    if drop_out:
        unit = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(),
                nn.ReLU()
                )
    else:
        unit = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )

    return unit

class PanUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PanUNet, self).__init__()
        self.rgb_conv = [
                nn.Conv2d(in_channels, 64, 3, 1, 1),
                conv_bn_relu(64, 128),
                conv_bn_relu(128, 256),
                conv_bn_relu(256, 512),
                conv_bn_relu(512, 512),
                conv_bn_relu(512, 512),
                conv_bn_relu(512, 512),
                ]

        self.panchro_conv = [
                nn.Conv2d(in_channels, 64, 3, 1, 1),
                conv_bn_relu(64, 128),
                conv_bn_relu(128, 256),
                conv_bn_relu(256, 512),
                conv_bn_relu(512, 512),
                conv_bn_relu(512, 512),
                conv_bn_relu(512, 512),
                conv_bn_relu(512, 512),
                ]

        self.deconv = [
                deconv_bn_relu(1024, 512, drop_out=True),
                deconv_bn_relu(1536, 512, drop_out=True),
                deconv_bn_relu(1536, 512, drop_out=True),
                deconv_bn_relu(1536, 512),
                deconv_bn_relu(1536, 256),
                deconv_bn_relu(768, 128),
                deconv_bn_relu(256, 64),
                nn.Conv2d(64, out_channels, 3, 1, 1)
                ]

    def forward(self, rgb, panchro):
        rgb_skips = []
        panchro_skips = []
        h1 = self.rgb_conv[0](rgb)
        for conv in self.rgb_conv[1:]:
            h1 = conv(h1)
            rgb_skips.append(h1)
        h2 = self.panchro_conv[0](panchro)
        for conv in self.panchro_conv[1:]:
            h2 = conv(h2)
            panchro_skips.append(h2)

        h = self.deconv[0](torch.cat((h1, h2), dim=1))
        for i, deconv in enumerate(self.deconv[1: -2]):
            h = torch.cat((h, rgb_skips[i], panchro_skips[i]), dim=1)
            h = deconv(h)
        h = self.deconvs[-2](toch.cat((h, panchro_skips[-1]), dim=1))
        h = self.deconv[-1](h)

        return h
