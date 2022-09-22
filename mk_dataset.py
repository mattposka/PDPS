# use saturation channel to preprocess leaf
# divide the large leaf image into several smaller training patches
# Author: Haomiao Ni

import os
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import numpy as np
import threading
from utils.transforms import im2vl
from itertools import chain

import scipy.stats as ss
import cv2
import glob
import preprocess as prep

# the main program of dividing leaf
def process_tumor_tif( imgfile, labelfile, image_number, validation_ratio, image_dir, label_dir, vimage_dir, vlabel_dir ):

    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    orig_r,orig_c,chans = img.shape

    label_img = cv2.imread(labelfile)
    label_img = label_img[:,:,:3]
    label_img = cv2.resize(label_img,(orig_r,orig_c))

    img = prep.makeSquare(img)
    label_img = prep.makeSquare(label_img)

    leaf_mask = prep.getLeafMask(img)

    left_cut,right_cut,top_cut,bot_cut = prep.getCropBounds(leaf_mask)
    half_side,row_mid,col_mid = prep.getCropBoundsSquare(left_cut,right_cut,top_cut,bot_cut)

    img = prep.cropSquare(img,half_side,row_mid,col_mid)
    label_img = prep.cropSquare(img,half_side,row_mid,col_mid)
    leaf_mask = prep.cropSquare(leaf_mask,half_side,row_mid,col_mid)

    img = prep.rmBackground(img,leaf_mask)

    img = cv2.resize(img,(512,512))
    label_img = cv2.resize(label_img,(512,512))

    # im2vl transforms label_img to 1s and 0s instead of 3d 0-255 values
    label_img = im2vl(label_img)
    label_img = prep.setEdgesToZero(label_img)

    image_name = None
    label_img_name = None
    if image_number % validation_ratio == 0:
        image_name = os.path.join(vimage_dir, os.path.basename(imgfile))
        labelname = os.path.basename(imgfile).replace( 'original',tumorname )
        label_img_name = os.path.join(vlabel_dir, labelname)
    else:
        image_name = os.path.join(image_dir, os.path.basename(imgfile))
        labelname = os.path.basename(imgfile).replace( 'original',tumorname )
        label_img_name = os.path.join(label_dir, labelname)

    print('\timage_name :',image_name)
    print('\tlabel_img_name :',label_img_name)
    cv2.imwrite(image_name,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    cv2.imwrite(label_img_name, label_img)


if __name__ == '__main__':
    root_pth = '/data/leaf_train/'
    tumorname = "green"
    version = "Sep22"

    validation_ratio = 8 # 1 out of every validation_ratio images will go to validation_dir

    img_pth = root_pth + "imgs_all"
    savepath = os.path.join(root_pth, tumorname, version )
    all_possible_files = glob.glob(img_pth+'/*.png')

    image_dir = os.path.join(savepath, 'images')
    label_dir = os.path.join(savepath, 'labels')

    validation_dir = os.path.join(savepath, 'validation')
    vimage_dir = os.path.join(validation_dir, 'images')
    vlabel_dir = os.path.join(validation_dir, 'labels')


    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    if not os.path.exists(vimage_dir):
        os.makedirs(vimage_dir)
    if not os.path.exists(vlabel_dir):
        os.makedirs(vlabel_dir)

    image_number = 1
    filenames = glob.glob(img_pth + '/*')
    for imgfile in filenames:
        if not "original" in imgfile:
            continue
        labelfile = imgfile.replace("original", tumorname)

        if labelfile not in all_possible_files:
            continue

        print('Processing :',imgfile)
        process_tumor_tif( imgfile, labelfile, image_number, validation_ratio, image_dir, label_dir, vimage_dir, vlabel_dir )
        image_number += 1

    traintxt = open(os.path.join(savepath, 'train.txt'), 'w')
    imglist = os.listdir(image_dir)
    for i in imglist:
        img_file_name = os.path.join(image_dir,i)
        label_file_name = os.path.join(label_dir,i.replace('original',tumorname))
        if os.path.exists(label_file_name):
            traintxt.writelines(img_file_name + ',' + label_file_name + '\n')
    traintxt.close()

    validationtxt = open(os.path.join(savepath, 'validation.txt'), 'w')
    imglist = os.listdir(vimage_dir)
    for i in imglist:
        img_file_name = os.path.join(vimage_dir,i)
        label_file_name = os.path.join(vlabel_dir,i.replace('original',tumorname))
        if os.path.exists(label_file_name):
            validationtxt.writelines(img_file_name + ',' + label_file_name + '\n')
    validationtxt.close()
