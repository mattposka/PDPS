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
def process_tif( file,filename,patch_image_dir,log,patch_size,overlap_size,ref_extent,ref_area,leaf_mask_dir='' ):
    print( '\nPreprocessing',filename )
    start = time()

    imgpath = os.path.join(patch_image_dir, filename[:-4])
    if not os.path.exists(imgpath):
        os.mkdir(imgpath)

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

    low_dim_img = Image.open(file)
    low_dim_img = low_dim_img.crop((rs,cs,re,ce))

    #low_dim_img = low_dim_img.resize((256,342))
    low_dim_img = low_dim_img.resize((patch_size,patch_size))
    #low_dim_img = low_dim_img.resize((patch_size,patch_size))
    ## HSV is Hue[0,179], Saturation[0,255], Value[0,255]
    low_hsv_img = low_dim_img.convert('HSV')
    _, low_s, _ = low_hsv_img.split()
    #print( 'low_s :',low_s )


    #############################################################################################################################
    #############################################################################################################################
    #############################################################################################################################
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
    # HSV is Hue[0,179], Saturation[0,255], Value[0,255]
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
    maskpth = os.path.join( leaf_mask_dir,(filename.replace('.png','')+'_mask.png') )
    plt.imsave( maskpth,leaf_mask )

    maskpth_p = maskpth.replace( '.png','.p' )
    p.dump( leaf_mask,open(maskpth_p,'wb') )
    #############################################################################################################################
    #############################################################################################################################
    #############################################################################################################################

    imagename = os.path.join(imgpath, filename[:-4] + '_Full.jpg')
    cv2.imwrite(imagename, img)

    stop = time()
    print('\tPreprocessing time : ' + str(stop - start))
    log.writelines('processing time : ' + str(stop - start) + '\n')


if __name__ == '__main__':
    # please check the following 2 parameters can be divided by 8
    # (It may be better that first parameter can be divided by 16).
    patch_size = 512
    overlap_size = 128
    ref_area = 10000
    ref_extent = 0.6
    root_pth = '/workspace/data/AutoPheno/'
    tumorname = "green"
    version = "200717"
    # img_pth1 = root_pth + "imgs"
    # img_pth2 = root_pth + "new_imgs_200527/lesion_training_set_2"
    # img_pth = (img_pth1, img_pth2)
    img_pth = [root_pth + "random_test_imgs"]
    savepath = os.path.join(root_pth, tumorname, version, "512_test_stride_64")
    logdirpth = os.path.join(root_pth, tumorname, version, 'log')
    if not os.path.exists(logdirpth):
        os.makedirs(logdirpth)
    logpath = osp.join(root_pth, tumorname, version, 'log', '512_test_stride_64_XY3c.log')

    images = os.path.join(savepath, 'images')
    labels = os.path.join(savepath, 'labels')

    if not os.path.exists(images):
        os.makedirs(images)
    if not os.path.exists(labels):
        os.makedirs(labels)

    NormalMask = os.path.join(labels, 'All_Normal_Mask.png')
    if not os.path.exists(NormalMask):
        NMask = Image.new('L', (patch_size, patch_size))
        NMask.save(NormalMask, 'PNG')

    log = open(logpath, 'w')

    total_start = time()

    for pth, dirs, filenames in chain.from_iterable(os.walk(path) for path in img_pth):
        for filename in filenames:
            # if not "original" in filename:
            #     continue
            # if not filename in test_imgs_list:
            #     continue
            if not filename.endswith('.png'):
                continue
            file = os.path.join(pth, filename)
            print(file)
            log.write(file + '\n')
            process_tif(file, filename, images, log, patch_size, overlap_size, ref_extent=ref_extent, ref_area=ref_area)

    total_stop = time()
    print("total processing time:", total_stop - total_start)
    log.writelines("total processing time : " + str(total_stop - total_start) + '\n')
    log.close()
