from __future__ import division
import torch
import torch.nn.functional as F
from torch import nn
from time import time

from .misc import initialize_weights


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),  # 512-3+1=510
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same'),  # 510-3+1=508
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # （508-2）/2+1=254
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size=3):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)

class _DeepSupervisionBlock(nn.Module):
    def __init__(self, channels ):
        super(_DeepSupervisionBlock, self).__init__()
        self.deep = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.deep(x)


# TODO add in deep supervision
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 64, kernel_size=5)
        self.enc2 = _EncoderBlock(64, 128, kernel_size=4)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64, kernel_size=4)
        self.dec1 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=5, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        self.deep0_0 = _DeepSupervisionBlock(512) # 64 -> 128
        self.deep0_1 = _DeepSupervisionBlock(512) # 128 -> 256
        self.deep0_2 = _DeepSupervisionBlock(512) # 256 -> 512
        self.deep1_0 = _DeepSupervisionBlock(256) # 128 -> 256
        self.deep1_1 = _DeepSupervisionBlock(256) # 256 -> 512
        self.deep2_0 = _DeepSupervisionBlock(128) # 256 -> 512
        # overall number of features added is 128+256+512=896

        initialize_weights(self)

    def forward(self, x):               # SAME Padding          |   No Padding
        enc1 = self.enc1(x)             # 512->256              |   512 -> 254 （512/2）-2
        enc2 = self.enc2(enc1)          # 256->128              |   254 -> 125
        enc3 = self.enc3(enc2)          # 128->64               |   125 -> 60
        enc4 = self.enc4(enc3)          # 64->32                |   60 -> 28
        center = self.center(enc4)      # 32->64                |   28 -> 48 ( ((28-2)-2)*2 )

        dec4 = self.dec4(torch.cat([center,enc4], 1) )  # 64->128
        dec3 = self.dec3(torch.cat([enc3,dec4], 1) )    # 128->256
        dec2 = self.dec2(torch.cat([enc2,dec3], 1) )    # 256->512

        deep0_0 = self.deep0_0(center)
        deep0_1 = self.deep0_1(deep0_0)
        deep0_2 = self.deep0_2(deep0_1)
        deep1_0 = self.deep1_0(dec4)
        deep1_1 = self.deep1_1(deep1_0)
        deep2_0 = self.deep2_0(dec3)

        dec1 = self.dec1(torch.cat([enc1,dec2,deep0_2,deep1_1,deep2_0], 1) )    # 512->512

        #dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=True)], 1))
        #dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear', align_corners=True)], 1))
        #dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=True)], 1))
        #dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=True)], 1))

        final = self.final(dec1)
        #return F.interpolate(final, int(x.size()[2]), mode='bilinear')
        return final


if __name__=='__main__':
    pass
