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


def divide_slide(img, filename, imgpath, thread_index, ranges, r_list, c_list, log, patch_size):
    for s in range(ranges[thread_index][0], ranges[thread_index][1]):
        shard = s
        r = r_list[shard]
        c = c_list[shard]

        try:
            topy = int(r-patch_size/2)
            buttomy = int(topy+patch_size)
            leftx = int(c-patch_size/2)
            rightx = int(leftx+patch_size)
            if topy<0 or leftx<0 or buttomy>img.shape[0] or rightx>img.shape[1]:
                print('Out of Index: ('+ str(leftx) + ',' + str(topy) + ") for " + filename)
                log.writelines('Out of Index: ('+ str(leftx) + ',' + str(topy) + ") for " + filename+'\n')
                continue

            imgarr = img[topy:buttomy, leftx:rightx]
            image = Image.fromarray(imgarr)
            image = image.convert('RGB')
        except:
            print("Can not read the point (" + str(leftx) + ',' + str(topy) + ") for " + filename)
            log.writelines("Can not read the point (" + str(leftx) + ',' + str(topy) + ") for " + filename + '\n')
            continue
        else:
            imagename = os.path.join(imgpath, filename[:-4] + '_' + str(leftx) + '_' + str(topy) + '.jpg')
            image.save(imagename, "JPEG")


# process_tif - 
def process_tif( file,filename,patch_image_dir,log,patch_size,overlap_size,ref_extent,ref_area,leaf_mask_dir='' ):
    print( '\nPreprocessing',filename )
    start = time()
    imgpath = os.path.join(patch_image_dir, filename[:-4])
    if not os.path.exists(imgpath):
        os.mkdir(imgpath)
    # else:
    #     return
    #img = scipy.misc.imread(file)
    low_dim_img = Image.open(file)
    ## HSV is Hue[0,179], Saturation[0,255], Value[0,255]
    low_hsv_img = low_dim_img.convert('HSV')
    _, low_s, _ = low_hsv_img.split()
    #print( 'low_s :',low_s )


    #############################################################################################################################
    #############################################################################################################################
    #############################################################################################################################
    img = cv2.imread(file)
    
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
    
    labeled_img = label(closed_mask, connectivity=2)

    # leaf area should be the second largest number, with background being the most common
    mode_label,count = ss.mode( labeled_img,axis=None )
    # remove the most common label here
    labeled_img_filtered = np.where( labeled_img==mode_label,np.nan,labeled_img )
    mode_label,count = ss.mode( labeled_img_filtered,axis=None,nan_policy='omit' )
    leaf_label = mode_label

    leaf_mask = np.where( labeled_img==leaf_label,True,False )
    low_s_bin = leaf_mask.copy()
    maskpth = os.path.join( leaf_mask_dir,('leaf_mask_'+filename) )
    plt.imsave( maskpth,leaf_mask )

    maskpth_p = maskpth.replace( '.png','.p' )
    p.dump( leaf_mask,open(maskpth_p,'wb') )
    #############################################################################################################################
    #############################################################################################################################
    #############################################################################################################################


    ## --OSTU threshold
    ## OSTU thresholding separates the foreground from the background
    #low_s_thre = filters.threshold_otsu(np.array(low_s))
    #print( '(ldTe) low_s_thre :',low_s_thre )
    #low_s_bin = low_s > low_s_thre  # row is y and col is x
    #print( '(ldTe) low_s_bin.shape :',low_s_bin.shape )
    #print( '(ldTe) low_s_bin :',low_s_bin )

    ## only keep the largest connected component
    ## TODO what does this mean?
    #low_s_bin = closing(low_s_bin, square(3))
    #labeled_img = label(low_s_bin, connectivity=2)

    #props = regionprops(labeled_img)
    #print( 'props :',props )
    #area_list = np.zeros(len(props))
    #for i, reg in enumerate(props):  # i+1 is the label
    #    area_list[i] = reg.area
    ## sort
    #area_list = area_list.argsort()[::-1]
    #label_id = -1
    #label_area = 0
    #extent = 0
    #for i in area_list:
    #    extent = props[i].extent
    #    if extent > ref_extent:
    #        label_id = i + 1
    #        label_area = props[i].area
    #        break
    #if label_id == -1 or label_area < ref_area:
    #    print("name:", file)
    #    print("extent:", extent)
    #    print("area", label_area)
    #assert label_id != -1, "fail to find the leaf region in pre-processing!" \
    #                       "try to REDUCE 'ref_extent' a bit"

    ## MP changed from assert label_area > ref_area to just print an error instead
    #####################################################################################################
    #if label_area > ref_area:
    #    print( '(ldTe) WARNING' )
    #    print( "(ldTe) failed to find the leaf region in pre-processing!\n Try to REDUCE 'ref_extent' a bit" )
    #####################################################################################################
    #low_s_bin = labeled_img == label_id

    #########################################################################################################################
    # divide low_s
    h = low_s.height
    w = low_s.width
    # h OR w = 500 + 450k + R
    h_k = (h - patch_size) // (patch_size - overlap_size)
    h_R = (h - patch_size) % (patch_size - overlap_size)
    if h_R <= overlap_size:
        h_flag = 0
    else:
        h_flag = 1
    w_k = (w - patch_size) // (patch_size - overlap_size)
    w_R = (w - patch_size) % (patch_size - overlap_size)
    if w_R <= overlap_size:
        w_flag = 0
    else:
        w_flag = 1

    h_list = []
    h_list.append(patch_size / 2)

    for i in range(1, h_k + 1):
        h_list.append(patch_size / 2 + (patch_size - overlap_size) * i)

    if not h_flag:
        if h_k >= 1:
            h_list.remove(patch_size / 2 + (patch_size - overlap_size) * h_k)
        h_list.append(h - patch_size / 2)
    else:
        h_list.append(h - patch_size / 2)

    w_list = []
    w_list.append(patch_size / 2)

    for j in range(1, w_k + 1):
        w_list.append(patch_size / 2 + (patch_size - overlap_size) * j)

    if not w_flag:
        if w_k >= 1:
            w_list.remove(patch_size / 2 + (patch_size - overlap_size) * w_k)
        w_list.append(w - patch_size / 2)
    else:
        w_list.append(w - patch_size / 2)

    [r_list, c_list] = np.meshgrid(h_list, w_list)
    r_list = r_list.flatten()
    c_list = c_list.flatten()
    # makes a grid of all of the patches x,y
    #########################################################################################################################

    tar_r_list = []
    tar_c_list = []

    for i in range(len(r_list)):
        r = r_list[i]
        c = c_list[i]
        topy = int(r - patch_size / 2)
        buttomy = int(r + patch_size / 2)
        leftx = int(c - patch_size / 2)
        rightx = int(c + patch_size / 2)
        low_patch = low_s_bin[topy:buttomy, leftx:rightx]
        if np.sum(low_patch):
            tar_r_list.append(r)
            tar_c_list.append(c)

    # patch start and end coords
    #########################################################################################################################

    r_list = tar_r_list
    c_list = tar_c_list

    # print(time() - start, 's')
    num_threads = 64
    num_patches = len(r_list)
    print('\tSplit into {} patches.'.format(num_patches) )
    log.writelines('num_patches : ' + str(num_patches) + '\n')

    spacing = np.linspace(0, num_patches, num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    img = cv2.cvtColor( img,cv2.COLOR_BGR2RGB )
    threads = []
    for thread_index in range(len(ranges)):
        args = (img, filename, imgpath, thread_index, ranges, r_list, c_list, log, patch_size)
        t = threading.Thread(target=divide_slide, args=args)
        t.setDaemon(True)
        threads.append(t)

    for t in threads:
        t.start()

    # Wait for all the threads to terminate.
    for t in threads:
        t.join()

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
