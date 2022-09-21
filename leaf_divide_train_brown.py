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
import glob

def process_tif( file,patch_size,mean=np.array((128,128,128)) ):
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


    #maskpth = os.path.join( leaf_mask_dir,(filename.replace('.png','')+'_mask.png') )
    #plt.imsave( maskpth,leaf_mask )

    #maskpth_p = maskpth.replace( '.png','.p' )
    #p.dump( leaf_mask,open(maskpth_p,'wb') )

    stop = time()
    #print('\tPreprocessing time : ' + str(stop - start))
    log.writelines('processing time : ' + str(stop - start) + '\n')

    return resized_img,normalized_img,leaf_mask

# the main program of dividing leaf
def process_tumor_tif(file, filename, labelfile, images, labels, isValid, log, rgb_sum, tumorname):
    start = time()
    saveNormal = True

    img = cv2.imread(file)
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
    img = img_square
    print('img.shape :',img.shape)

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
    print('leaf_mask.shape :',leaf_mask.shape)

    left_cut = 0
    right_cut = 0
    column_sum = np.sum(leaf_mask,axis=0)
    print('column_sum.shape :',column_sum.shape)
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
    print('row_sum.shape :',row_sum.shape)
    for i in range(r):
        if row_sum[i] > 0:
            top_cut = max(0,i-1)
            break
    for i in range(r,0,-1):
        if row_sum[i-1] > 0:
            bot_cut = i
            break
   
    #img = img[top_cut:bot_cut,left_cut:right_cut,:]
    #leaf_mask = leaf_mask[top_cut:bot_cut,left_cut:right_cut]
    print('leaf_mask.shape :',leaf_mask.shape)
    #r,c = leaf_mask.shape
    #for i in range(r):
    #    for j in range(c):
    #        if leaf_mask[i,j] == False:
    #            img[i,j,:] = 0
##################################################################################################

    height = bot_cut - top_cut
    width = right_cut - left_cut 
    print('height :',height)
    print('width :',width)

    r,c,_ = img.shape
    square_size = max(height,width)
    ss2 = int(square_size / 2)
    r2 = int(top_cut + height / 2)
    c2 = int(left_cut + width / 2)
    print('r2 :',r2)
    print('c2 :',c2)
    print('ss2 :',ss2)
    img = img[r2-ss2:r2+ss2,c2-ss2:c2+ss2]
    leaf_mask = leaf_mask[r2-ss2:r2+ss2,c2-ss2:c2+ss2]
    r,c = leaf_mask.shape
    print('leaf_mask.shape :',leaf_mask.shape)
    for i in range(r):
        for j in range(c):
            if leaf_mask[i,j] == False:
                img[i,j,:] = 0

    img = cv2.resize(img,(512,512))

    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    print( 'img_rgb.shape :',img_rgb.shape )
    tot_r = np.sum(img_rgb[:,:,0])
    tot_g = np.sum(img_rgb[:,:,1])
    tot_b = np.sum(img_rgb[:,:,2])
    mean_r = tot_r/(512*512)
    mean_g = tot_g/(512*512)
    mean_b = tot_b/(512*512)
    print('mr,mg,mb :',mean_r,mean_g,mean_b)
    rgb_sum += np.array((mean_r,mean_g,mean_b))


    label_img = scipy.misc.imread(labelfile)
    label_img = label_img[:,:,:3]
    label_img = cv2.resize(label_img,(orig_r,orig_c))

    r,c,chans = label_img.shape
    label_square = None
    if r > c:
        label_square = np.zeros((r,r,chans))
        diff = r - c
        label_square[:,int(diff/2):-int(diff/2),:] = label_img
    else:
        label_square = np.zeros((c,c,chans))
        diff = c - r
        label_square[int(diff/2):-int(diff/2),:,:] = label_img
    label_img = label_square
    print('label_img.shape :',label_img.shape)
##################################################################################################
    #label_img = label_img[top_cut:bot_cut,left_cut:right_cut]
##################################################################################################

    #r,c,_ = label_img.shape
    #square_size = np.max([r,c])
    #ss2 = square_size/2
    #r2 = r/2
    #rs = int(r2-ss2)
    #re = int(r2+ss2)
    #c2 = c/2
    #cs = int(c2-ss2)
    #ce = int(c2+ss2)
    #label_img = label_img[rs:re,cs:ce]
    label_img = label_img[r2-ss2:r2+ss2,c2-ss2:c2+ss2]

    label_img = scipy.misc.imresize(label_img, (512,512) )

    # im2vl transforms label_img to 1s and 0s instead of 3d 0-255 values
    label_img = im2vl(label_img)
    label_img[:1,:] = 0
    label_img[-1:,:] = 0
    label_img[:,:1] = 0
    label_img[:,-1:] = 0


    if isValid == False:
        imagename = os.path.join(Tumordir, filename[:-4] + '_Full.jpg')
        cv2.imwrite(imagename,img)

        labelname = filename.replace( 'original',tumorname )
        label_img_name = os.path.join(labels, labelname[:-4] + '_Full.png')
        cv2.imwrite(label_img_name, label_img)
    if isValid == True:
        imagename = os.path.join(vtumor_dir, filename[:-4] + '_Full.jpg')
        cv2.imwrite(imagename,img)

        labelname = filename.replace( 'original',tumorname )
        label_img_name = os.path.join(vlabels, labelname[:-4] + '_Full.png')
        cv2.imwrite(label_img_name, label_img)

    stop = time()
#    print('processing time : ' + str(stop - start))
    log.writelines('processing time : ' + str(stop - start) + '\n')
    return rgb_sum


if __name__ == '__main__':
    root_pth = '/data/leaf_train/'
    tumorname = "brown"
    version = "Sep22"
    # clear test image list, i.e. using all the labeled images
    test_imgs_list = []

    rgb_sum = np.array((0,0,0),dtype=np.float32)

    validation_ratio = 8 # 1 out of every validation_ratio images will go to validation_dir

    # img_pth1 = root_pth + "imgs"
    # img_pth2 = root_pth + "new_imgs_200527/lesion_training_set_2"
    img_pth = [root_pth + "imgs_all"]
    all_possible_files = glob.glob(img_pth[0]+'/*.png')
    savepath = osp.join(root_pth, tumorname, version, 'Full')
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
    #for pth, dirs, filenames in chain.from_iterable(os.walk(path) for path in img_pth):
    #    for filename in filenames:
    #        if not "original" in filename:
    #            continue
    #        if filename in test_imgs_list:
    #            continue
    #        file = os.path.join(pth, filename)
    #        labelfile = file.replace("original", tumorname)
#
#            if labelfile not in all_possible_files:
#                continue

#            if image_number % validation_ratio == 0:
#                isValid = True
#            else:
#                isValid = False
#            rgb_sum = process_tumor_tif(file, filename, labelfile, images, labels, isValid, log, rgb_sum, tumorname)
#            image_number += 1

    total_stop = time()
#    print("total processing time:", total_stop - total_start)
    log.writelines("total processing time : " + str(total_stop - total_start) + '\n')
    log.close()

    #Tumor = os.path.join(images, tumorname)
    tumorlist = os.listdir(Tumordir)
    print( 'tumorlist :',tumorlist )
    print( 'lables :',labels )
    tumortxt = open(os.path.join(savepath, 'train.txt'), 'w')
    print( 'savepath :',savepath )
    for t in tumorlist:
        tumorfilename = os.path.join(Tumordir, t)
        label_file = t.replace( 'jpg','png' )
        label_file = label_file.replace( 'original',tumorname )
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
        label_file = label_file.replace( 'original',tumorname )
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
