import numpy as np
from time import time
import threading
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import cv2
import scipy.stats as ss

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
    leaf_mask = np.where( labeled_mask==leaf_label,1,0 )
    leaf_mask = np.array(leaf_mask,dtype='uint8')
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

def annotationsToLabel(label_img):
    img_tmp = np.ones(label_img.shape[:2], dtype=np.uint8)
    neg_msk = np.all(label_img == [255, 255, 255], axis=2)
    img_tmp[neg_msk] = 0
    return img_tmp

def setEdgesToZero(label_img):
    label_img[:1,:] = 0
    label_img[-1:,:] = 0
    label_img[:,:1] = 0
    label_img[:,-1:] = 0
    return label_img

def normalizeImage(img):
    mean = np.mean(img,axis=(0,1))
    std = np.std(img,axis=(0,1))
    return ( img - mean ) / std

def process_tif( file ):

    img = cv2.imread(file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = makeSquare(img)

    leaf_mask = getLeafMask(img)

    left_cut,right_cut,top_cut,bot_cut = getCropBounds(leaf_mask)
    half_side,row_mid,col_mid = getCropBoundsSquare(left_cut,right_cut,top_cut,bot_cut)

    img = cropSquare(img,half_side,row_mid,col_mid)
    leaf_mask = cropSquare(leaf_mask,half_side,row_mid,col_mid)

    img = rmBackground(img,leaf_mask)

    #print('max(leaf_mask) :',np.max(leaf_mask))
    #print('shape(leaf_mask) :',leaf_mask.shape)
    img = cv2.resize(img,(512,512))
    leaf_mask = cv2.resize(leaf_mask,(512,512))

    normalized_img = normalizeImage(img)

    total_reshaped_pixels = 4 * half_side * half_side
    resize_ratio = total_reshaped_pixels/(512*512)


    #normalized_img_pth = file.replace('.png','_norm.png')
    #cv2.imwrite(normalized_img_pth,np.float32(cv2.cvtColor(np.float32(normalized_img),cv2.COLOR_RGB2BGR)))
    return img,normalized_img,leaf_mask,resize_ratio,half_side,row_mid,col_mid

def quick_process_tif(file,leaf_mask,row_mid,col_mid,half_side):

    img = cv2.imread(file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h,w,c = img.shape

    img = makeSquare(img)
    img = cropSquare(img,half_side,row_mid,col_mid)

    img = cv2.resize(img,(512,512))
    img = rmBackground(img,leaf_mask)

    total_reshaped_pixels = 4*half_side*half_side
    resize_ratio = total_reshaped_pixels/(512*512)

    normalized_img = normalizeImage(img)

    #normalized_img_pth = file.replace('.png','_norm.png')
    #cv2.imwrite(normalized_img_pth,np.float32(cv2.cvtColor(np.float32(normalized_img),cv2.COLOR_RGB2BGR)))
    return img,normalized_img,resize_ratio