# use saturation channel to preprocess leaf
# divide the large leaf image into several smaller training patches
# Author: Haomiao Ni

import os
import os.path as osp
from skimage import filters
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import numpy as np
from time import time
from PIL import Image
import threading
from scipy.sparse import coo_matrix
import scipy
from utils.transforms import im2vl
from utils.mk_datasets import test_imgs_list
from itertools import chain

from skimage.measure import label, regionprops
from skimage.morphology import closing, square

import scipy.stats as ss
import cv2

# the main program of dividing leaf
def process_tumor_tif(file, filename, maskfile, circlefile, images, labels, isValid, log, rgb_sum  ):
    start = time()
    saveNormal = True
    print('Preparing :',file)

    imagename1 = os.path.join(Tumordir, filename[:-4] + '_Full.npy')
    imagename0 = os.path.join(vtumor_dir, filename[:-4] + '_Full.npy')
    if os.path.exists(imagename1) or os.path.exists(imagename0):
        return rgb_sum

    img = cv2.imread(file)
    original_r,original_c,_ = img.shape
    #print('img.shape :',img.shape )
    h,w,c = img.shape
    img_empty = np.zeros( (max(h,w),max(h,w),c),dtype=img.dtype )
    diff = max(h,w)-min(h,w)
    start_idx = int(diff/2)
    if h==max(h,w):
        img_empty[:,start_idx:start_idx+w,:] = img
    else:
        img_empty[start_idx:start_idx+h,:,:] = img
    img = img_empty
    #print('img010.shape :',img.shape )

##################################################################################################
# From process_tif - Get the background
##################################################################################################
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

##################################################################################################
# Resize Image to remove as much background as possible
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

    #print('height :',height )
    #print('width :',width )
    #print('center_h :',center_h)
    #print('center_w :',center_w)
    #print('half_side:',half_side )

    #
    #print('img00.shape :',img.shape)
    img = img[center_h-half_side:center_h+half_side,center_w-half_side:center_w+half_side,:]
    r,c,_ = img.shape
    if r == 0 or c == 0:
        return rgb_sum
    #print('img000.shape :',img.shape)
    leaf_mask = leaf_mask[center_h-half_side:center_h+half_side,center_w-half_side:center_w+half_side]
    r,c = leaf_mask.shape
    for i in range(c):
        for j in range(r):
            if leaf_mask[i,j] == False:
                img[i,j,:] = 0
##################################################################################################

    r,c,_ = img.shape
    #square_size = np.min([r,c])
    #ss2 = square_size/2
    #r2 = r/2
    #rs = int(r2-ss2)
    #re = int(r2+ss2)
    #c2 = c/2
    #cs = int(c2-ss2)
    #ce = int(c2+ss2)
    #img = img[rs:re,cs:ce]

    img = cv2.resize(img,(512,512))

    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    tot_r = np.sum(img_rgb[:,:,0])
    tot_g = np.sum(img_rgb[:,:,1])
    tot_b = np.sum(img_rgb[:,:,2])
    mean_r = tot_r/(512*512)
    mean_g = tot_g/(512*512)
    mean_b = tot_b/(512*512)
    #print('rgb_sum :',rgb_sum)
    rgb_sum += np.array((mean_r,mean_g,mean_b))


    mask = scipy.misc.imread(maskfile)
    mask = mask[:,:,:3]
##################################################################################################
    #print( 'mask0.shape :',mask.shape )
    mask = scipy.misc.imresize(mask,(original_r,original_c))
    #print( 'mask1.shape :',mask.shape )

    h,w,c = mask.shape

    mask_empty = np.zeros( (max(h,w),max(h,w),c) )
    diff = max(h,w)-min(h,w)
    start_idx = int(diff/2)
    if h==max(h,w):
        mask_empty[:,start_idx:start_idx+w,:] = mask
    else:
        mask_empty[start_idx:start_idx+h,:,:] = mask
    mask = mask_empty
    #print( 'mask2.shape :',mask.shape )

    #mask = mask[top_cut:bot_cut,left_cut:right_cut,:]
    mask = mask[center_h-half_side:center_h+half_side,center_w-half_side:center_w+half_side,:]
    #print( 'mask3.shape :',mask.shape )
##################################################################################################

    #r,c,_ = mask.shape
    #square_size = np.min([r,c])
    #ss2 = square_size/2
    #r2 = r/2
    #rs = int(r2-ss2)
    #re = int(r2+ss2)
    #c2 = c/2
    #cs = int(c2-ss2)
    #ce = int(c2+ss2)
    #mask = mask[rs:re,cs:ce]

    mask = scipy.misc.imresize(mask, (512,512) )
    #print( 'mask4.shape :',mask.shape )

    # im2vl transforms mask to 1s and 0s instead of 3d 0-255 values
    mask = im2vl(mask)
    mask[:1,:] = 0
    mask[-1:,:] = 0
    mask[:,:1] = 0
    mask[:,-1:] = 0
    
    pos_pixels = np.sum(mask)
    bg_pixels = int(512*512) - int(pos_pixels)

###################################################################################
    circle = scipy.misc.imread(circlefile)
    #print('circle0.shape :',circle.shape)
    #print('circle0.mean :',np.mean(circle) )
    circle = circle[:,:,:3]
##################################################################################################
    circle = scipy.misc.imresize(circle,(original_r,original_c))
    #circle = circle[top_cut:bot_cut,left_cut:right_cut,:]
    #print('circle0.shape :',circle.shape)

    h,w,c = circle.shape

    circle_empty = np.zeros( (max(h,w),max(h,w),c) )
    diff = max(h,w)-min(h,w)
    start_idx = int(diff/2)
    if h==max(h,w):
        circle_empty[:,start_idx:start_idx+w,:] = circle
    else:
        circle_empty[start_idx:start_idx+h,:,:] = circle
    circle = circle_empty

    circle = circle[center_h-half_side:center_h+half_side,center_w-half_side:center_w+half_side,:]
    #print('circle1.shape :',circle.shape)
    #print('circle1.mean :',np.mean(circle) )
##################################################################################################
    #r,c,_ = circle.shape
    #square_size = np.min([r,c])
    #ss2 = square_size/2
    #r2 = r/2
    #rs = int(r2-ss2)
    #re = int(r2+ss2)
    #c2 = c/2
    #cs = int(c2-ss2)
    #ce = int(c2+ss2)
    #circle = circle[rs:re,cs:ce]

    circle = scipy.misc.imresize(circle, (512,512) )
    circle = circle[:,:,:3]
    #print('circle2.shape :',circle.shape)
    #print('circle2.mean :',np.mean(circle) )

    # im2vl transforms circle to 1s and 0s instead of 3d 0-255 values
    circle = im2vl(circle)
    circle[:1,:] = 0
    circle[-1:,:] = 0
    circle[:,:1] = 0
    circle[:,-1:] = 0
    #print('circle3.shape :',circle.shape)
    #print('circle3.mean :',np.mean(circle) )
    
    if np.max(circle) != 1:
        print('Something Wrong?')
    
    r,c = circle.shape
    circle = circle.reshape(r,c,1)

    cir_img = np.concatenate([img,circle],axis=-1)
    
    blank_channel = np.zeros((r,c,1))
    imgWBC = np.concatenate([img,blank_channel],axis=-1)
###################################################################################


    if isValid == False:
        #imagename = os.path.join(Tumordir, filename[:-4] + '_Full.png')
        #cv2.imwrite(imagename,imgWBC)

        imagename = os.path.join(Tumordir, filename[:-4] + '_Full.npy')
        np.save(imagename,imgWBC)

        #circlename = os.path.join(Tumordir, filename[:-4] + '_FullC.png')
        #cv2.imwrite(circlename,cir_img)

        circlename = os.path.join(Tumordir, filename[:-4] + '_FullC.npy')
        #print('np.max(cir_img[:,:,3]) :',np.max(cir_img[:,:,3]))
        np.save(circlename,cir_img)

        #labelname = filename.replace( 'original','green' )
        #maskname = os.path.join(labels, labelname[:-4] + '_Full.png')
        #cv2.imwrite(maskname, mask)

        labelname = filename.replace( 'original','green' )
        maskname = os.path.join(labels, labelname[:-4] + '_Full.npy')
        np.save(maskname, mask)

    if isValid == True:
        #imagename = os.path.join(vtumor_dir, filename[:-4] + '_Full.png')
        #cv2.imwrite(imagename,imgWBC)

        imagename = os.path.join(vtumor_dir, filename[:-4] + '_Full.npy')
        np.save(imagename,imgWBC)

        #circlename = os.path.join(vtumor_dir, filename[:-4] + '_FullC.png')
        #cv2.imwrite(circlename,cir_img)

        circlename = os.path.join(vtumor_dir, filename[:-4] + '_FullC.npy')
        np.save(circlename,cir_img)

        #labelname = filename.replace( 'original','green' )
        #maskname = os.path.join(vlabels, labelname[:-4] + '_Full.png')
        #cv2.imwrite(maskname, mask)

        labelname = filename.replace( 'original','green' )
        maskname = os.path.join(vlabels, labelname[:-4] + '_Full.npy')
        np.save(maskname, mask)

    stop = time()
#    print('processing time : ' + str(stop - start))
    log.writelines('processing time : ' + str(stop - start) + '\n')
    return rgb_sum, pos_pixels, bg_pixels


if __name__ == '__main__':
    root_pth = '/data/leaf_train/'
    tumorname = "green"
    version = "Sep2021"
    # clear test image list, i.e. using all the labeled images
    test_imgs_list = []

    rgb_sum = np.array((0,0,0),dtype=np.float32)

    validation_ratio = 8 # 1 out of every validation_ratio images will go to validation_dir

    # img_pth1 = root_pth + "imgs"
    # img_pth2 = root_pth + "new_imgs_200527/lesion_training_set_2"
    img_pth = [root_pth + "imgs_all"]
    savepath = osp.join(root_pth, tumorname, version, 'Circle')
    logdirpth = osp.join(root_pth, tumorname, version, 'log')
    if not os.path.exists(logdirpth):
        os.makedirs(logdirpth)
    logpath = osp.join(root_pth, tumorname, version, 'log', 'Full_XY3c.log')

    images = os.path.join(savepath, 'images')
    labels = os.path.join(savepath, 'labels')

    validation_dir = os.path.join(savepath, 'validation')
    vimages = os.path.join(validation_dir, 'images')
    vlabels = os.path.join(validation_dir, 'labels')
    vtumor_dir = os.path.join(vimages, tumorname)


    if not os.path.exists(images):
        os.makedirs(images)
    if not os.path.exists(labels):
        os.makedirs(labels)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    if not os.path.exists(vimages):
        os.makedirs(vimages)
    if not os.path.exists(vlabels):
        os.makedirs(vlabels)
    if not os.path.exists(vtumor_dir):
        os.makedirs(vtumor_dir)

    Tumordir = os.path.join(images, tumorname)
    if not os.path.exists(Tumordir):
        os.makedirs(Tumordir)

    log = open(logpath, 'w')

    total_start = time()

    image_number = 1
    pos_pixels_tot = 0
    bg_pixels_tot = 0
    for pth, dirs, filenames in chain.from_iterable(os.walk(path) for path in img_pth):
        for filename in filenames:
            if not "original" in filename:
                continue
            if filename in test_imgs_list:
                continue
            file = os.path.join(pth, filename)
            maskfile = file.replace("original", tumorname)
            circlefile = file.replace("original", tumorname+'C')

            if image_number % validation_ratio == 0:
                isValid = True
            else:
                isValid = False

            rgb_sum,pos_pixels,bg_pixels = process_tumor_tif(file, filename, maskfile, circlefile, images, labels, isValid, log, rgb_sum )

            pos_pixels_tot = pos_pixels_tot + pos_pixels
            print('pos_pixels :',pos_pixels_tot)
            bg_pixels_tot = bg_pixels_tot + bg_pixels
            print('bg_pixels :',bg_pixels_tot)
            image_number += 1

    total_stop = time()
#    print("total processing time:", total_stop - total_start)
    log.writelines("total processing time : " + str(total_stop - total_start) + '\n')
    log.close()

    Tumor = os.path.join(images, tumorname)
    tumorlist = os.listdir(Tumor)
    print( 'tumorlist :',tumorlist )
    print( 'lables :',labels )
    tumortxt = open(os.path.join(savepath, 'train.txt'), 'w')
    print( 'savepath :',savepath )
    for t in tumorlist:
        tumorfilename = os.path.join(Tumor, t)
        label_file = t.replace( 'jpg','png' )
        label_file = label_file.replace( 'original','green' )
###############################################################
        if 'C' in label_file:
            label_file = label_file.replace('C','')
###############################################################
        labelname = os.path.join(labels, label_file)
        print( 'labelname :',labelname )
        if os.path.exists(labelname):
            tumortxt.writelines(tumorfilename + ' ' + labelname + '\n')
    tumortxt.close()

    valtxt = open(os.path.join(savepath, 'validation.txt'), 'w')
    vallist = os.listdir(vtumor_dir)
    for t in vallist:
        valfilename = os.path.join(vtumor_dir, t)
        label_file = t.replace( 'jpg','png' )
        label_file = label_file.replace( 'original','green' )
        labelname = os.path.join(vlabels, label_file)
        print( 'vlabelname :',labelname )
        if os.path.exists(labelname):
            valtxt.writelines(valfilename + ' ' + labelname + '\n')
    valtxt.close()

    print( 'mean_rgb :',rgb_sum/(image_number-1) )

    #Normal = os.path.join(images, 'normal')
    #normallist = os.listdir(Normal)
    #normaltxt = open(os.path.join(savepath, 'normal.txt'), 'w')
    #labelname = os.path.join(labels, 'All_Normal_Mask.png')
    #for n in normallist:
    #    normalname = os.path.join(Normal, n)
    #    normaltxt.writelines(normalname + ' ' + labelname + '\n')
    #normaltxt.close()

