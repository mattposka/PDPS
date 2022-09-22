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


def makeSquare(img):
    datatype = img.dtype
    r,c,chans = img.shape
    orig_r,orig_c,chans = img.shape
    img_square = None
    if r > c:
        img_square = np.zeros((r,r,chans),dtype=datatype)
        diff = r - c
        img_square[:,int(diff/2):-int(diff/2),:] = img
    else:
        img_square = np.zeros((c,c,chans),dtype=datatype)
        diff = c - r
        img_square[int(diff/2):-int(diff/2),:,:] = img
    return img_square

def getLeafMask(img):
    # HSV is Hue[0,179], Saturation[0,255], Value[0,255]
    hsv_img = cv2.cvtColor( img,cv2.COLOR_RGB2HSV )
    hue = hsv_img[:,:,0]
    sat = hsv_img[:,:,1]
    val = hsv_img[:,:,2]
    
    ret_sat,thresh_sat = cv2.threshold( sat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU ) 
    ret_hue,thresh_hue = cv2.threshold( hue,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU ) 
    mask = cv2.bitwise_and( thresh_hue,thresh_sat,mask=None )
    
    # only keep the largest connected component
    closed_mask = closing( mask,square(3) )
    labeled_mask = label(closed_mask, connectivity=2)

    # leaf area should be the second largest number, with background being the most common
    mode_label,count = ss.mode( labeled_mask,axis=None )
    # remove the most common label here
    labeled_mask_filtered = np.where( labeled_mask==mode_label,np.nan,labeled_mask )
    mode_label,count = ss.mode( labeled_mask_filtered,axis=None,nan_policy='omit' )
    leaf_label = mode_label
    leaf_mask = np.where( labeled_mask==leaf_label,True,False )
    return leaf_mask

def getCropBounds(leaf_mask):
    r,c = leaf_mask.shape
    left_cut = 0
    right_cut = 0
    column_sum = np.sum(leaf_mask,axis=0)
    for i in range(c):
        if column_sum[i] > 0:
            left_cut = max(0,i-1)
            break
    for i in range(c,0,-1):
        if column_sum[i-1] > 0:
            right_cut = i
            break
    top_cut = 0
    bot_cut = 0
    row_sum = np.sum(leaf_mask,axis=1)
    for i in range(r):
        if row_sum[i] > 0:
            top_cut = max(0,i-1)
            break
    for i in range(r,0,-1):
        if row_sum[i-1] > 0:
            bot_cut = i
            break
    return left_cut,right_cut,top_cut,bot_cut

def getCropBoundsSquare(left_cut,right_cut,top_cut,bot_cut):
    height = bot_cut - top_cut
    width = right_cut - left_cut 

    square_size = max(height,width)
    half_side = int(square_size / 2)
    row_mid = int(top_cut + height / 2)
    col_mid = int(left_cut + width / 2)
    return half_side,row_mid,col_mid

def cropSquare(img,half_side,row_mid,col_mid):
    return img[row_mid-half_side:row_mid+half_side,col_mid-half_side:col_mid+half_side]

def rmBackground(img,mask):
    r,c = mask.shape
    for i in range(r):
        for j in range(c):
            if mask[i,j] == False:
                img[i,j,:] = 0
    return img

def setEdgesToZero(label_img):
    label_img[:1,:] = 0
    label_img[-1:,:] = 0
    label_img[:,:1] = 0
    label_img[:,-1:] = 0
    return label_img

# the main program of dividing leaf
def process_tumor_tif( imgfile, labelfile, image_number, validation_ratio, image_dir, label_dir, vimage_dir, vlabel_dir ):

    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    orig_r,orig_c,chans = img.shape

    # Lode labeled img and resize to same size as input image
    label_img = cv2.imread(labelfile)
    label_img = label_img[:,:,:3]
    label_img = cv2.resize(label_img,(orig_r,orig_c))

    # Make image square
    img = makeSquare(img)
    label_img = makeSquare(label_img)
    #print('img.shape :',img.shape)

    leaf_mask = getLeafMask(img)
    #print('leaf_mask.shape :',leaf_mask.shape)

    left_cut,right_cut,top_cut,bot_cut = getCropBounds(leaf_mask)
    half_side,row_mid,col_mid = getCropBoundsSquare(left_cut,right_cut,top_cut,bot_cut)

    img = cropSquare(img,half_side,row_mid,col_mid)
    label_img = cropSquare(img,half_side,row_mid,col_mid)
    leaf_mask = cropSquare(leaf_mask,half_side,row_mid,col_mid)

    img = rmBackground(img,leaf_mask)

    img = cv2.resize(img,(512,512))
    label_img = cv2.resize(label_img,(512,512))

    # im2vl transforms label_img to 1s and 0s instead of 3d 0-255 values
    label_img = im2vl(label_img)
    label_img = setEdgesToZero(label_img)

    image_name = None
    label_img_name = None

    if image_number % validation_ratio == 0:
        image_name = os.path.join(image_dir, os.path.basename(imgfile))
        labelname = filename.replace( 'original',tumorname )
        label_img_name = os.path.join(label_dir, os.path.basename(labelfile))
    else:
        image_name = os.path.join(vimage_dir, os.path.basename(imgfile))
        labelname = filename.replace( 'original',tumorname )
        label_img_name = os.path.join(vlabel_dir, os.path.basename(labelfile))

    cv2.imwrite(image_name,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    cv2.imwrite(label_img_name, label_img)


if __name__ == '__main__':
    root_pth = '/data/leaf_train/'
    tumorname = "brown"
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
