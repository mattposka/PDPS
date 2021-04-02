from __future__ import division
import torch
import torch.nn.functional as F
from torch import nn
from time import time

from .misc import initialize_weights


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),  # 512-3+1=510
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),  # 510-3+1=508
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
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)

class _LeftLeg(nn.Module):
    def __init__(self, dropout=False):
        super(_LeftLeg, self).__init__()
        self.leftleg = nn.Sequential( nn.MaxPool2d(kernel_size=2, stride=2) )

    def forward(self, x):
        return self.leftleg(x)

class _RightLeg(nn.Module):
    def __init__(self, dropout=False):
        super(_RightLeg, self).__init__()
        self.rightleg = nn.Sequential( nn.MaxPool2d(kernel_size=2, stride=2) )

    def forward(self, x):
        return self.rightleg(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.ll1 = _LeftLeg()
        self.ll2 = _LeftLeg()
        #self.ll3 = _LeftLeg()
        self.rl1 = _RightLeg()
        self.rl2 = _RightLeg()
        #self.rl3 = _RightLeg()
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(65, 128)
        self.enc3 = _EncoderBlock(129, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(XX, num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    # shape (batches x channels x rows x cols)
    # TODO why not just use padding=same?
    def forward(self, x):
        ll1 = self.ll1(x)
        ll2 = self.ll2(ll1)
        #ll3 = self.ll3(ll2)
        enc1 = self.enc1(x)  # ( batch_size x 64 channels x 512 -> 254 512/2）-2 )
        enc2 = self.enc2(torch.cat([enc1,F.interpolate(ll1,enc2.size()[2:],mode='bilinear',alight_corners=True)],1))  # (batch_size x 128 channels x 254 -> 125 )
        enc3 = self.enc3(torch.cat([enc2,F.interpolate(ll2,enc3.size()[2:],mode='bilinear',alight_corners=True)],1))  # (batch_size x 256 channels x 125 ->60 )
        enc4 = self.enc4(enc3)  #( batch_size x 512 channels x 60 -> 28 )
        center = self.center(enc4)  # (batch_size x 512 channels x 56 )
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear', align_corners=True)], 1))
        rl1 = self.rl1( dec3 )
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=True)], 1))
        rl2 = self.rl1( dec2 )
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=True)], 1))
        final = self.final(dec1)
        return F.interpolate(final, int(x.size()[2]), mode='bilinear')


if __name__=='__main__':
    pass
