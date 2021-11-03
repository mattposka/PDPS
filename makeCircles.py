import glob
import numpy as np
import cv2
from skimage.measure import label

<<<<<<< HEAD
img_dir = '/data/leaf_train/imgs_all/'

img_files = glob.glob(img_dir+'*')
# change greenC to something else later!!
already_labeled = glob.glob(img_dir+'*C.png')

label_files = []
for f in img_files:
    if 'green.png' in f:
        found = False
        for lf in already_labeled:
            if f.replace('.png','C.png') == lf:
                found = True
                print( 'found' )
        if found == False:
            label_files.append(f)
=======
img_dir = '/home/mposka/data/mattLabeled/'

img_files = glob.glob(img_dir+'*')

label_files = []
for f in img_files:
    if 'green' in f:
        label_files.append(f)
>>>>>>> d539fa91231f386be49cd3600e9f7c8c6233e733

for lf in label_files:
    print( 'Working on : ',lf )

    # Read Label Image
    lab_img = cv2.imread(lf)

    # Convert Label Image to 1D
    lab_arr = np.array(lab_img)    
    row,col,chan = lab_arr.shape

    lab_arr_1d = np.zeros(shape=(row,col))
    for i in range(chan):
        lab_arr_1d[lab_arr[:,:,i]!=255]=1
    lab_arr_1d = label(lab_arr_1d) # Label the Different Lesions for FindContours

    # Find Contours for opencv functions
    lab_1d = np.asarray( lab_arr_1d,dtype=np.uint8 )
    contours,h = cv2.findContours(lab_1d,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )

    # Find the Largest Inscribed Circle for each lesion
    centers = []
    rads = []
    for c in range(len(contours)):
        raw_dist = np.empty(lab_1d.shape)
        for i in range(row):
            for j in range(col):
                raw_dist[i,j] = cv2.pointPolygonTest(contours[c],(j,i),True)
    
        minVal,maxVal,_,maxDistPt = cv2.minMaxLoc(raw_dist)
        centers.append(maxDistPt)
        rads.append(int(maxVal))

    # Make new Label Img File
    new_lab_img = np.zeros(lab_img.shape,lab_img.dtype)
    new_lab_img += 255
    for c in range(len(centers)):
        cv2.circle(new_lab_img,centers[c],rads[c],(0,0,255),-1)

    cv2.imwrite(lf.replace('.png','C.png'),new_lab_img)
