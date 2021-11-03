from __future__ import division
import torch
import torch.nn.functional as F
from torch import nn
from time import time

<<<<<<< HEAD
import kornia as K
from kornia import morphology as morph

=======
>>>>>>> d539fa91231f386be49cd3600e9f7c8c6233e733
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


class UNetDICE(nn.Module):
    def __init__(self, num_classes):
        super(UNetDICE, self).__init__()
        self.enc1 = _EncoderBlock(4, 96, ksize=3)

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
        self.dense1 = nn.Conv2d(64, 64, kernel_size=1)
        self.dense2 = nn.Conv2d(64, 16, kernel_size=1)
<<<<<<< HEAD
        self.dense3 = nn.Conv2d(16, num_classes, kernel_size=1)
        self.dense4 = nn.Conv2d(2, 1, kernel_size=1)

        self.kernelE = torch.tensor([[1,1,1],[1,1,1],[1,1,1]],dtype=torch.float).cuda()
        #self.erode1 = morph.erosion(self.kernelE)
        #self.erode2 = morph.erosion(self.kernelE)
        #self.erode3 = morph.erosion(self.kernelE)
        #self.erode4 = morph.erosion(self.kernelE)
        #self.erode5 = morph.erosion(self.kernelE)
        #self.erode6 = morph.erosion(self.kernelE)

=======
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
>>>>>>> d539fa91231f386be49cd3600e9f7c8c6233e733
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
        #centerD6 = self.centerD6(torch.cat([enc4,F.interpolate(enc4D,enc4.size()[2:],mode='bilinear',align_corners=True)],1))
        #print('centerD6.size() :',centerD6.size() ) 

        #dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec4 = self.dec4(torch.cat([center, \
            F.interpolate(centerD2, center.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(centerD4, center.size()[2:], mode='bilinear', align_corners=True), \
            #F.interpolate(centerD6, center.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(enc4D, center.size()[2:], mode='bilinear', align_corners=True), \
            ],1))
        #print('dec4.size() :',dec4.size() ) 

        dec4D = self.dec4D(torch.cat([center, \
            F.interpolate(centerD2, center.size()[2:], mode='bilinear', align_corners=True), \
            F.interpolate(centerD4, center.size()[2:], mode='bilinear', align_corners=True), \
            #F.interpolate(centerD6, center.size()[2:], mode='bilinear', align_corners=True), \
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
        dense1 = self.dense1(dec1)
<<<<<<< HEAD
        #print('dense1.size() :',dense1.size())
        dense2 = self.dense2(dense1)
        #print('dense2.size() :',dense2.size())
        dense3 = self.dense3(dense2)
        #print('final.size() :',final.size())
        dense4 = self.dense4(dense3)
        #print('dense4.size() :',dense4.size())
        #print('x.size() :',x.size())
        #final_sig = self.final_sig(final2) 
        #final_sig2 = F.interpolate(final_sig, int(x.size()[2]), mode='bilinear')
        #final1 = F.interpolate(dense4, int(x.size()[2]), mode='bilinear')

        #self.kernelE = torch.tensor([[1,1,1],[1,1,1],[1,1,1]])
        #self.erode1 = morph.erosion(self.kernelE)
        #self.erode2 = morph.erosion(self.kernelE)
        #self.erode3 = morph.erosion(self.kernelE)
        #self.erode4 = morph.erosion(self.kernelE)
        #self.erode5 = morph.erosion(self.kernelE)
        #self.erode6 = morph.erosion(self.kernelE)

#        self.kernelE = torch.tensor([[1,1,1],[1,1,1],[1,1,1]]).cuda()
        erode1 = morph.erosion(dense4,self.kernelE).cuda()
        erode2 = morph.erosion(erode1,self.kernelE).cuda()
        erode3 = morph.erosion(erode2,self.kernelE).cuda()
        erode4 = morph.erosion(erode3,self.kernelE).cuda()
        erode5 = morph.erosion(erode4,self.kernelE).cuda()
        erode6 = morph.erosion(erode5,self.kernelE).cuda()

        #erode1 = self.erode1(final1)
        #erode2 = self.erode2(erode1)
        #erode3 = self.erode3(erode2)
        #erode4 = self.erode4(erode3)
        #erode5 = self.erode5(erode4)
        #erode6 = self.erode6(erode5)
        #print('erode6.shape :',erode6.shape )


        final1 = F.interpolate(dense4, int(x.size()[2]), mode='bilinear')
        ob,oc,oh,ow = final1.shape
        final1 = torch.reshape(final1,(ob,oh,ow))

        final2 = F.interpolate(erode6, int(x.size()[2]), mode='bilinear')
        final2 = torch.reshape(final2,(ob,oh,ow))
      
        #return F.interpolate(final, int(x.size()[2]), mode='bilinear')
        return final1, final2
=======
        dense2 = self.dense2(dense1)
        final = self.final(dense2)
        return F.interpolate(final, int(x.size()[2]), mode='bilinear')
>>>>>>> d539fa91231f386be49cd3600e9f7c8c6233e733


if __name__=='__main__':
    pass
