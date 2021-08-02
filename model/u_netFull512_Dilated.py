from __future__ import division
import torch
import torch.nn.functional as F
from torch import nn
from time import time

from .misc import initialize_weights


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, dilation=1, padding=0, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=ksize, dilation=dilation, padding=padding),  # 512-3+1=510
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=ksize, dilation=dilation, padding=padding),  # 510-3+1=508
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
    def __init__(self, in_channels, middle_channels, out_channels, ksize=3, dilation=1, padding=0):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=ksize, dilation=dilation, padding=padding),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=ksize, dilation=dilation, padding=padding),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNetFull512_Dilated(nn.Module):
    def __init__(self, num_classes):
        super(UNetFull512_Dilated, self).__init__()
        self.enc1 = _EncoderBlock(3, 96, ksize=3)

        self.enc2 = _EncoderBlock(96, 100, ksize=3 )
        self.enc2D = _EncoderBlock(96, 56, ksize=3, dilation=2, padding=1)

        self.enc3 = _EncoderBlock(156, 170)
        self.enc3D = _EncoderBlock(156, 86, dilation=2, padding=1)

        self.enc4 = _EncoderBlock(256, 384, dropout=True)
        self.enc4D = _EncoderBlock(256, 128, dropout=True, dilation=2, padding=1)

        #self.center = _DecoderBlock(512, 1024, 512, dilation=1)
        self.center = _DecoderBlock(512, 640, 320, dilation=1)
        self.centerD2 = _DecoderBlock(512, 256, 128, dilation=2, padding=1)
        self.centerD4 = _DecoderBlock(512, 128, 64, dilation=4, padding=3)
        #self.centerD6 = _DecoderBlock(512, 128, 64, dilation=6, padding=5)

        #self.dec4 = _DecoderBlock(1024, 512, 256)
        #self.dec4D = _DecoderBlock(1024, 512, 256,dilation=4)
        self.dec4 = _DecoderBlock(1024, 256, 128)
        self.dec4D = _DecoderBlock(1024, 256, 128,dilation=2,padding=1)

        #self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec3 = _DecoderBlock(512, 128, 64)
        self.dec3D = _DecoderBlock(512, 128, 64, dilation=2,padding=1)

        #self.dec2 = _DecoderBlock(284, 128, 64)
        self.dec2 = _DecoderBlock(284, 64, 32)
        self.dec2D = _DecoderBlock(284, 64, 32, dilation=2,padding=1)

        self.dec1 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)  # 512 254 （512/2）-2
        #print('enc1.size() :',enc1.size() ) 
        enc2 = self.enc2(enc1)  # 254 125
        #print('enc2.size() :',enc2.size() ) 
        enc2D = self.enc2D(enc1)  # 254 125
        #print('enc2D.size() :',enc2D.size() ) 
        enc3 = self.enc3(torch.cat([enc2,F.interpolate(enc2D,enc2.size()[2:],mode='bilinear',align_corners=True)],1))  # 60
        #print('enc3.size() :',enc3.size() ) 
        enc3D = self.enc3D(torch.cat([enc2,F.interpolate(enc2D,enc2.size()[2:],mode='bilinear',align_corners=True)],1))  # 60
        #print('enc3D.size() :',enc3D.size() ) 
        enc4 = self.enc4(torch.cat([enc3,F.interpolate(enc3D,enc3.size()[2:],mode='bilinear',align_corners=True)],1))
        #print('enc4.size() :',enc4.size() ) 
        enc4D = self.enc4D(torch.cat([enc3,F.interpolate(enc3D,enc3.size()[2:],mode='bilinear',align_corners=True)],1))
        #print('enc4D.size() :',enc4D.size() ) 
        #enc4 = self.enc4(enc3)  # 28
        #center = self.center(enc4)  # 48
        center = self.center(torch.cat([enc4,F.interpolate(enc4D,enc4.size()[2:],mode='bilinear',align_corners=True)],1))
        #print('center.size() :',center.size() ) 
        centerD2 = self.centerD2(torch.cat([enc4,F.interpolate(enc4D,enc4.size()[2:],mode='bilinear',align_corners=True)],1))
        #print('centerD2.size() :',centerD2.size() ) 
        centerD4 = self.centerD4(torch.cat([enc4,F.interpolate(enc4D,enc4.size()[2:],mode='bilinear',align_corners=True)],1))
        #print('centerD4.size() :',centerD4.size() ) 
        centerD6 = self.centerD6(torch.cat([enc4,F.interpolate(enc4D,enc4.size()[2:],mode='bilinear',align_corners=True)],1))
        #print('centerD6.size() :',centerD6.size() ) 

        #dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec4 = self.dec4(torch.cat([center, \
            F.interpolate(centerD2, center.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(centerD4, center.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(centerD6, center.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc4D, center.size()[2:], mode='bilinear', align_corners=True), \
            ],1))
        #print('dec4.size() :',dec4.size() ) 

        dec4D = self.dec4D(torch.cat([center, \
            F.interpolate(centerD2, center.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(centerD4, center.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(centerD6, center.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc4D, center.size()[2:], mode='bilinear', align_corners=True), \
            ],1))
        #print('dec4D.size() :',dec4D.size() ) 

        dec3 = self.dec3(torch.cat([dec4, \
            F.interpolate(dec4D, dec4.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc3, dec4.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc3D, dec4.size()[2:], mode='bilinear', align_corners=True), \
            ], 1))
        #print('dec3.size() :',dec3.size() ) 

        dec3D = self.dec3D(torch.cat([dec4, \
            F.interpolate(dec4D, dec4.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc3, dec4.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc3D, dec4.size()[2:], mode='bilinear', align_corners=True), \
            ], 1))
        #print('dec3D.size() :',dec3D.size() ) 

        dec2 = self.dec2(torch.cat([dec3, \
            F.interpolate(dec3D, dec3.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc2D, dec3.size()[2:], mode='bilinear', align_corners=True), \
            ], 1))
        #print('dec2.size() :',dec2.size() ) 

        dec2D = self.dec2D(torch.cat([dec3, \
            F.interpolate(dec3D, dec3.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc2D, dec3.size()[2:], mode='bilinear', align_corners=True), \
            ], 1))
        #print('dec2D.size() :',dec2D.size() ) 

        #dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec1 = self.dec1(torch.cat([dec2, \
            F.interpolate(dec2D, dec2.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=True), \
            ], 1))
        #print('dec1.size() :',dec1.size() ) 

        #dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=True)], 1))
        final = self.final(dec1)
        return F.interpolate(final, int(x.size()[2]), mode='bilinear')


if __name__=='__main__':
    pass
