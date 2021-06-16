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
import glob

import scipy.stats as ss
import cv2


# the sub-program of dividing leaf
def divide_Tumor_slide(img, mask, filename, images, labels, thread_index, ranges, sparse_s_bin,
                       saveNormal, log):
    print( 'img :',img )
    print( 'mask :',mask )
    print( 'filename :',filename )
    print( 'images :',images )
    print( 'labels :',labels )
    print( 'thread_index :',thread_index )
    print( 'ranges :',ranges )
    print( 'sparse_s_bin :',sparse_s_bin )

    labelname = filename.replace( 'original','green' )

    Normaldir = os.path.join(images, 'normal')
    Tumordir = os.path.join(images, tumorname)
    for s in range(ranges[thread_index][0], ranges[thread_index][1]):
        shard = s
        r = sparse_s_bin.row[shard]
        c = sparse_s_bin.col[shard]
        try:
            topy = int(r - 512 / 2)
            buttomy = topy + 512
            leftx = int(c - 512 / 2)
            rightx = leftx + 512
            if topy < 0 or leftx < 0 or buttomy > img.shape[0] or rightx > img.shape[1]:
                continue
            image = Image.fromarray(img[topy:buttomy, leftx:rightx])
            image = image.convert('RGB')
            print( 'mask.shape :',mask.shape )
            array_mask = mask[topy:buttomy, leftx:rightx]
            print( 'array_mask :',array_mask )
            print( 'array_mask.shape :',array_mask.shape )
            print( 'np.max(array_mask) :',np.max(array_mask) )
        except:
            print("Can not read the point (" + str(leftx) + ',' + str(topy) + ") for " + filename)
            log.writelines("Can not read the point (" + str(leftx) + ',' + str(topy) + ") for " + filename + '\n')
            continue
        else:
            tumorid = np.argwhere(array_mask != 0)
            IsTumor = (len(tumorid) > 0)
            if IsTumor:  # Tumor, need to save mask
                imagename = os.path.join(Tumordir, filename[:-4] + '_' + str(leftx) + '_' + str(topy) + '.jpg')
                image.save(imagename, "JPEG")
                #maskname = os.path.join(labels, filename[:-4] + '_' + str(leftx) + '_' + str(topy) + '.png')
                #TODO check maskname
                maskname = os.path.join(labels, labelname[:-4] + '_' + str(leftx) + '_' + str(topy) + '.png')
                #leaf_mask = Image.fromarray(array_mask)
                #leaf_mask = leaf_mask.convert('L')
                #print( 'maskname :',maskname )
                #leaf_mask.save(maskname, 'PNG')

                leaf_mask = Image.fromarray(array_mask)
                #leaf_mask = leaf_mask.convert('L')
                leaf_mask.save(maskname, 'PNG')
                print( 'maskname :',maskname )
                #scipy.misc.imsave(array_mask, 'PNG')
            else:  # normal
                if saveNormal == True:
                    imagename = os.path.join(Normaldir, filename[:-4] + '_' + str(leftx) + '_' + str(topy) + '.jpg')
                    image.save(imagename, "JPEG")


# the main program of dividing leaf
def process_tumor_tif(file, filename, maskfile, images, labels, log, ref_extent, ref_area):
    start = time()
    saveNormal = True

    filename = file.split('/')[-1]

    img = scipy.misc.imread(file)
    low_dim_img = Image.open(file)
    low_hsv_img = low_dim_img.convert('HSV')
    _, low_s, _ = low_hsv_img.split()

    #///////////////////////////////////////////////////////////////////////////////////////
    #///////////////////////////////////////////////////////////////////////////////////////
    img = cv2.imread(file)
    img = cv2.cvtColor( img,cv2.COLOR_BGR2RGB )
    
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
    
    labeled_img = label(closed_mask, connectivity=2)

    # leaf area should be the second largest number, with background being the most common
    mode_label,count = ss.mode( labeled_img,axis=None )
    # remove the most common label here
    labeled_img_filtered = np.where( labeled_img==mode_label,np.nan,labeled_img )
    mode_label,count = ss.mode( labeled_img_filtered,axis=None,nan_policy='omit' )
    leaf_label = mode_label

    leaf_mask = np.where( labeled_img==leaf_label,True,False )
    low_s_bin = leaf_mask.copy()
    print( 'low_s_bin.shape :',low_s_bin.shape )
    #maskpth = os.path.join( leaf_mask_dir,('leaf_mask_'+filename) )
    #plt.imsave( maskpth,leaf_mask )

    #maskpth_p = maskpth.replace( '.png','.p' )
    #p.dump( leaf_mask,open(maskpth_p,'wb') )
    #///////////////////////////////////////////////////////////////////////////////////////
    #///////////////////////////////////////////////////////////////////////////////////////


    ##########################################################################################
    ##########################################################################################
    ## --OSTU threshold
    #low_s_thre = filters.threshold_otsu(np.array(low_s))
    #low_s_bin = low_s > low_s_thre  # row is y and col is x

    ## only keep the largest connected component
    #low_s_bin = closing(low_s_bin, square(3))
    #labeled_img = label(low_s_bin, connectivity=2)

    ## low_s_bin = labeled_img == np.argmax(np.bincount(labeled_img.flat)[1:]) + 1

    #props = regionprops(labeled_img)
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
    #    print("extent:", extent)
    #    print("area", ref_area)

    #assert label_id != -1, "fail to find the leaf region in pre-processing!" \
    #                       "try to REDUCE 'ref_extent' a bit"
    #assert label_area > ref_area, "fail to find the leaf region in pre-processing!" \
    #                              "try to REDUCE 'ref_extent' a bit"
    #low_s_bin = labeled_img == label_id
    ##########################################################################################
    ##########################################################################################

    mask = scipy.misc.imread(maskfile)
    print( 'mask.shape( orig ):',mask.shape )
    # some masks are too big for some reason
    if ( mask.shape[2] ) > 3 :
        mask = mask[:,:,:3]
    mask = scipy.misc.imresize(mask, low_s_bin.shape)

    # mask transform
    print( 'maskfile :',maskfile )
    print( 'mask.shape :',mask.shape )
    #TODO fix this 
    mask = im2vl(mask)
    print( 'mask.shape (again) :',mask.shape )
    print( 'np.max( mask ) :',np.max( mask ) )
    print( 'np.min( mask ) :',np.min( mask ) )
    mask[:2,:] = 0
    mask[-2:,:] = 0
    mask[:,:2] = 0
    mask[:,-2:] = 0

    sample_bin = np.zeros(low_s_bin.shape, dtype=np.int)
    for r in range(0, low_s_bin.shape[0], 128):
        for c in range(0, low_s_bin.shape[1], 128):
            if low_s_bin[r, c] != 0:
                sample_bin[r, c] = 1

    print(time() - start, 's')
    num_threads = 1
    num_patches = np.sum(sample_bin)
    sparse_s_bin = coo_matrix(sample_bin)
    assert num_patches == len(sparse_s_bin.data)
    print('num_patches : ', num_patches)
    log.writelines('num_patches : ' + str(num_patches) + '\n')

    spacing = np.linspace(0, len(sparse_s_bin.data), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    threads = []
    for thread_index in range(len(ranges)):
        args = (img, mask, filename, images, labels, thread_index, ranges, sparse_s_bin, saveNormal, log)
        t = threading.Thread(target=divide_Tumor_slide, args=args)
        t.setDaemon(True)
        threads.append(t)

    for t in threads:
        t.start()

    # Wait for all the threads to terminate.
    for t in threads:
        t.join()

    stop = time()
    print('processing time : ' + str(stop - start))
    log.writelines('processing time : ' + str(stop - start) + '\n')


if __name__ == '__main__':
    root_pth = '/home/mposka/data/leaf_train/'
    tumorname = "green"
    version = "jun21"
    # clear test image list, i.e. using all the labeled images
    test_imgs_list = []
    ref_area = 10000
    ref_extent = 0.6
    # img_pth1 = root_pth + "imgs"
    # img_pth2 = root_pth + "new_imgs_200527/lesion_training_set_2"
    img_pth = [root_pth + "imgs_new"]
    savepath = osp.join(root_pth, tumorname, version, '512_train_stride_128')
    logdirpth = osp.join(root_pth, tumorname, version, 'log')
    if not os.path.exists(logdirpth):
        os.makedirs(logdirpth)
    logpath = osp.join(root_pth, tumorname, version, 'log', '512_train_stride_128_XY3c.log')

    images = os.path.join(savepath, 'images')
    labels = os.path.join(savepath, 'labels')
    print( 'images :',images )
    print( 'labels :',labels )

    if not os.path.exists(images):
        os.makedirs(images)
    if not os.path.exists(labels):
        os.makedirs(labels)

    Normaldir = os.path.join(images, 'normal')
    if not os.path.exists(Normaldir):
        os.makedirs(Normaldir)
    Tumordir = os.path.join(images, tumorname)
    if not os.path.exists(Tumordir):
        os.makedirs(Tumordir)

    NormalMask = os.path.join(labels, 'All_Normal_Mask.png')
    if not os.path.exists(NormalMask):
        NMask = Image.new('L', (512, 512))
        NMask.save(NormalMask, 'PNG')

    log = open(logpath, 'w')

    total_start = time()

    print( '00' )
    print( 'img_pth :',img_pth )
    # img_pth = (img_pth1, img_pth2)
    #for pth, dirs, filenames in chain.from_iterable(os.walk(path) for path in img_pth):
    filelist = glob.glob( str(img_pth[0]) + '/*' )
    print( 'filelist :',filelist )
    for filename in glob.glob( img_pth[0] + '/*' ):
            #print( 'pth :',pth )
        #print( 'dirs :',dirs )
        #print( 'filenames :',filenames )
        print( '33' )
        #for filename in filenames:
###########################################################################################################################################3

        print( 'filename :',filename )
        if not "original" in filename:
            print( '11' )
            continue
        if filename in test_imgs_list:
            print( '22' )
            continue
    #file = os.path.join(pth, filename)
        file = filename
        print( 'file :',file )
        maskfile = file.replace("original", tumorname)
        print('Tumor', file)
        print( 'maskfile :',maskfile )
        process_tumor_tif(file, filename, maskfile, images, labels, log, ref_area=ref_area, ref_extent=ref_extent)

###########################################################################################################################################3
#            print( 'filename :',filename )
#            if not "original" in filename:
#                print( '11' )
#                continue
#            if filename in test_imgs_list:
#                print( '22' )
#                continue
#            file = os.path.join(pth, filename)
#            print( 'file :',file )
#            maskfile = file.replace("original", tumorname)
#            print('Tumor', file)
#            print( 'maskfile :',maskfile )
#            process_tumor_tif(file, filename, maskfile, images, labels, log, ref_area=ref_area, ref_extent=ref_extent)

    total_stop = time()
    print("total processing time:", total_stop - total_start)
    log.writelines("total processing time : " + str(total_stop - total_start) + '\n')
    log.close()

    Tumor = os.path.join(images, tumorname)
    tumorlist = os.listdir(Tumor)
    print( 'tumorlist :',tumorlist )
    print( 'lables :',labels )
    tumortxt = open(os.path.join(savepath, tumorname+'.txt'), 'w')
    print( 'savepath :',savepath )
    for t in tumorlist:
        tumorfilename = os.path.join(Tumor, t)
        label_file = t.replace( 'jpg','png' )
        label_file = label_file.replace( 'original','green' )
        labelname = os.path.join(labels, label_file)
        print( 'labelname :',labelname )
        if os.path.exists(labelname):
            tumortxt.writelines(tumorfilename + ' ' + labelname + '\n')
    tumortxt.close()

    Normal = os.path.join(images, 'normal')
    normallist = os.listdir(Normal)
    normaltxt = open(os.path.join(savepath, 'normal.txt'), 'w')
    labelname = os.path.join(labels, 'All_Normal_Mask.png')
    for n in normallist:
        normalname = os.path.join(Normal, n)
        normaltxt.writelines(normalname + ' ' + labelname + '\n')
    normaltxt.close()

