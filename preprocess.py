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
import scipy.stats as ss
import pickle as p


# takes input image file image
#TODO 
# maybe instead of process_tif for everything, instead we have a :
# squareAndCenterInputImage()
# rmbackground()
# reshape()/resize()
# getLeafMask()
# normalize()
# inverse_normalize()

# convert the image into a square by putting black bars on 
# the top+bot or left+right
def squareAndCenterInputImage(file):

    img = cv2.imread(file)

    h,w,c = img.shape

    original_h,original_w,original_c = img.shape
    img_square = np.zeros( (max(h,w),max(h,w),c),dtype=img.dtype )
    diff = max(h,w)-min(h,w)
    start_idx = int(diff/2)
    if h==max(h,w):
        img_square[:,start_idx:start_idx+w,:] = img
    else:
        img_square[start_idx:start_idx+h,:,:] = img
    img = img_square

    return img

def removeBackground(img):

    # first filter out the background using HSV thresholds
    hsv_img = cv2.cvtColor( img,cv2.COLOR_BGR2HSV )
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

    #maskpth_p = maskpth.replace( '.png','.p' )
    #p.dump( leaf_mask,open(maskpth_p,'wb') )

###################################################################################################
###################################################################################################
    original_r,original_c = leaf_mask.shape
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
   
    height = bot_cut-top_cut
    width = right_cut-left_cut
    large_side = max(height,width)
    half_side = int( large_side / 2 )    
    center_h = int(top_cut + (height/2))
    center_w = int(left_cut + (width/2))

    print('height :',height )
    print('width :',width )
    print('center_h :',center_h)
    print('center_w :',center_w)
    print('half_side:',half_side )

    
    print('img00.shape :',img.shape)
    print('np.max(img00) :',np.max(img) )
    img = img[center_h-half_side:center_h+half_side,center_w-half_side:center_w+half_side,:]
    r,c,_ = img.shape
    if r == 0 or c == 0:
        return rgb_sum
    print('img000.shape :',img.shape)
    print('np.max(img000) :',np.max(img) )
    leaf_mask = leaf_mask[center_h-half_side:center_h+half_side,center_w-half_side:center_w+half_side]
    print('leaf_mask.shape :',leaf_mask.shape )
    print('np.sum(leaf_mask) :',np.sum(leaf_mask))
    print('np.max(leaf_mask) :',np.max(leaf_mask))
    np.save('leaf_mask.npy',leaf_mask)
    r,c = leaf_mask.shape
    for i in range(c):
        for j in range(r):
            if leaf_mask[i,j] == False:
                img[i,j,:] = 0
    print('img020.shape :',img.shape)
    print('np.max(img020) :',np.max(img) )


def process_tif( file,patch_size,mean=np.array((128,128,128)) ):
    filename = file.split('/')[-1]
    #print( '\nPreprocessing',filename )
    start = time()

    img = cv2.imread(file)

    #print('img.shape :',img.shape )
    #print('np.max(img) :',np.max(img) )

    h,w,c = img.shape
    original_h,original_w,original_c = img.shape

    total_original_pixels = original_h*original_w

    # Adding black bars to make the image square
    img_square = np.zeros( (max(h,w),max(h,w),c),dtype=img.dtype )
    diff = max(h,w)-min(h,w)
    start_idx = int(diff/2)
    if h==max(h,w):
        img_square[:,start_idx:start_idx+w,:] = img
    else:
        img_square[start_idx:start_idx+h,:,:] = img
    img = img_square

    #print('img.shape :',img.shape )
    #print('np.max(img) :',np.max(img) )

    #r,c,_ = img.shape
    #square_size = np.min([r,c])
    #ss2 = square_size/2
    #r2 = r/2
    #rs = int(r2-ss2)
    #re = int(r2+ss2)
    #c2 = c/2
    #cs = int(c2-ss2)
    #ce = int(c2+ss2)
    #img = img[rs:re,cs:ce]
    #img = cv2.resize(img,(patch_size,patch_size))
    #resized_img = img


    #imgpath = os.path.join(resized_image_dir, filename[:-4])
    #if not os.path.exists(imgpath):
    #    os.mkdir(imgpath)
    #imagename = os.path.join(imgpath, filename[:-4] + '_Full.jpg')
    #cv2.imwrite(imagename, img)

    # HSV is Hue[0,179], Saturation[0,255], Value[0,255]
    #hsv_img = cv2.cvtColor( resized_img,cv2.COLOR_BGR2HSV )
    hsv_img = cv2.cvtColor( img,cv2.COLOR_BGR2HSV )
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

    #maskpth_p = maskpth.replace( '.png','.p' )
    #p.dump( leaf_mask,open(maskpth_p,'wb') )

###################################################################################################
###################################################################################################
    original_r,original_c = leaf_mask.shape
    r,c = leaf_mask.shape
    # leaf mask should be the same shape as the input image

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
   
    height = bot_cut-top_cut
    width = right_cut-left_cut
    large_side = max(height,width)
    half_side = int( large_side / 2 )    
    center_h = int(top_cut + (height/2))
    center_w = int(left_cut + (width/2))

    #print('height :',height )
    #print('width :',width )
    #print('center_h :',center_h)
    #print('center_w :',center_w)
    ##print('half_side:',half_side )

    
    #print('img00.shape :',img.shape)
    #print('np.max(img00) :',np.max(img) )
    img = img[center_h-half_side:center_h+half_side,center_w-half_side:center_w+half_side,:]

    total_reshaped_pixels = 4*half_side*half_side

    r,c,_ = img.shape
    if r == 0 or c == 0:
        return rgb_sum
    #print('img000.shape :',img.shape)
    #print('np.max(img000) :',np.max(img) )
    leaf_mask = leaf_mask[center_h-half_side:center_h+half_side,center_w-half_side:center_w+half_side]
    #print('leaf_mask.shape :',leaf_mask.shape )
    #print('np.sum(leaf_mask) :',np.sum(leaf_mask))
    #print('np.max(leaf_mask) :',np.max(leaf_mask))
    np.save('leaf_mask.npy',leaf_mask)
    r,c = leaf_mask.shape
    for i in range(c):
        for j in range(r):
            if leaf_mask[i,j] == False:
                img[i,j,:] = 0
    #print('img020.shape :',img.shape)
    #print('np.max(img020) :',np.max(img) )

    resized_img = cv2.resize(img,(512,512))
    #print('resized.shape :',resized_img.shape)
    #print('np.max(resized_img) :',np.max(resized_img) )
    leaf_mask = np.array( leaf_mask,dtype=img.dtype )
    leaf_mask_resized = cv2.resize(leaf_mask,(512,512))

    #TODO check that this works
    #print('mean :',mean)
    r = mean[0]
    g = mean[1]
    b = mean[2]
    bgr_mean = np.array((b,g,r))
    normalized_img = np.array( resized_img, dtype='float64')
    #print('normalized1.shape :',normalized_img.shape)
    #print('np.max(normalized1_img) :',np.max(normalized_img) )
    if np.max(img[:,:,:3]) > 1:
        normalized_img[:,:,:3] = normalized_img[:,:,:3]/255.0
        #print('normalized.shape :',normalized_img.shape)
        #print('np.max(normalized_img) :',np.max(normalized_img) )
#    normalized_img = img - mean
#    normalized_img = normalized_img / mean
##################################################################################################
###################################################################################################
    resize_ratio = total_reshaped_pixels/(512*512)
    #resize_ratio = max(original_h,original_w)*max(original_h,original_w)/large_side*large_side

    stop = time()
    #print('\tPreprocessing time : ' + str(stop - start))

    return resized_img,normalized_img,leaf_mask,resize_ratio,center_h,center_w,half_side

def quick_process_tif(file, patch_size, leaf_mask,center_h,center_w,half_side):
    filename = file.split('/')[-1]
    img = cv2.imread(file)
    h,w,c = img.shape

    # Adding black bars to make the image square
    img_square = np.zeros( (max(h,w),max(h,w),c),dtype=img.dtype )
    diff = max(h,w)-min(h,w)
    start_idx = int(diff/2)
    if h==max(h,w):
        img_square[:,start_idx:start_idx+w,:] = img
    else:
        img_square[start_idx:start_idx+h,:,:] = img
    img = img_square

    img = img[center_h-half_side:center_h+half_side,center_w-half_side:center_w+half_side,:]

    total_reshaped_pixels = 4*half_side*half_side

    #leaf_mask = leaf_mask[center_h-half_side:center_h+half_side,center_w-half_side:center_w+half_side]
    np.save('leaf_mask.npy',leaf_mask)

    #r,c = leaf_mask.shape
    #for i in range(c):
    #    for j in range(r):
    #        if leaf_mask[i,j] == False:
    #            img[i,j,:] = 0
    for i in range(3):
        img[:,:,i] = np.where(leaf_mask==True,img[:,:,i],0)

    resized_img = cv2.resize(img,(512,512))
    leaf_mask = np.array( leaf_mask,dtype=img.dtype )
    leaf_mask_resized = cv2.resize(leaf_mask,(512,512))

    normalized_img = np.array( resized_img, dtype='float64')
    if np.max(img[:,:,:3]) > 1:
        normalized_img[:,:,:3] = normalized_img[:,:,:3]/255.0

    resize_ratio = total_reshaped_pixels/(512*512)

    return resized_img,normalized_img,leaf_mask,resize_ratio
