# Dividinz testing patches
# Author: Haomiao Ni
import os
import os.path as osp
from skimage import filters
import numpy as np
from time import time
from PIL import Image
import threading
from utils.mk_datasets import test_imgs_list
import scipy
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from itertools import chain
# MP added to test
import cv2
import matplotlib.pyplot as plt
import scipy.stats as ss
import pickle as p

# process_tif - 
def process_tif( file,log,patch_size,mean=np.array((128,128,128)) ):
    filename = file.split('/')[-1]
    #print( '\nPreprocessing',filename )
    start = time()

    img = cv2.imread(file)

    r,c,_ = img.shape
    square_size = np.min([r,c])
    ss2 = square_size/2
    r2 = r/2
    rs = int(r2-ss2)
    re = int(r2+ss2)
    c2 = c/2
    cs = int(c2-ss2)
    ce = int(c2+ss2)
    img = img[rs:re,cs:ce]
    img = cv2.resize(img,(patch_size,patch_size))
    resized_img = img

    #TODO check that this works
    #print('mean :',mean)
    r = mean[0]
    g = mean[1]
    b = mean[2]
    bgr_mean = np.array((b,g,r))
    normalized_img = img - mean
    normalized_img = normalized_img / mean

    #imgpath = os.path.join(resized_image_dir, filename[:-4])
    #if not os.path.exists(imgpath):
    #    os.mkdir(imgpath)
    #imagename = os.path.join(imgpath, filename[:-4] + '_Full.jpg')
    #cv2.imwrite(imagename, img)

    # HSV is Hue[0,179], Saturation[0,255], Value[0,255]
    hsv_img = cv2.cvtColor( resized_img,cv2.COLOR_BGR2HSV )
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

    #maskpth = os.path.join( leaf_mask_dir,(filename.replace('.png','')+'_mask.png') )
    #plt.imsave( maskpth,leaf_mask )

    #maskpth_p = maskpth.replace( '.png','.p' )
    #p.dump( leaf_mask,open(maskpth_p,'wb') )

    stop = time()
    #print('\tPreprocessing time : ' + str(stop - start))
    log.writelines('processing time : ' + str(stop - start) + '\n')

    return resized_img,normalized_img,leaf_mask
