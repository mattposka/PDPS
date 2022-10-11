import numpy as np
import pandas as pd

import imageio as io
from skimage.measure import label, regionprops
from skimage.color import label2rgb,gray2rgb
from skimage.segmentation import watershed
import cv2
import os
import scipy.ndimage as ndi

def watershedSegStack(seg_stack,num_lesions,postprocess_dir,cam_num):

    seg_stack = seg_stack.copy()
    imgs,rows,cols = seg_stack.shape
    sum_stack = np.sum(seg_stack,axis=(0)) / imgs
    seg_stack_save_pth = os.path.join(postprocess_dir,'cam' + str(cam_num) + 'segstack.png')
    cv2.imwrite( seg_stack_save_pth,255.0*gray2rgb(sum_stack) )

    bin_img = np.where(sum_stack > 0, True, False)
    distance = ndi.distance_transform_edt(bin_img)

    min_lesions_over_threshold = float('inf')
    max_lesions_under_threshold = 0
    enough_lesions_found = False
    best_labels = None
    threshold = 1.0
    for i in range(20):

        threshold -= 0.05
        starting_regions = np.where(sum_stack >= threshold, 1, 0)
        labels = label(starting_regions)

        label_map_save_pth = os.path.join(postprocess_dir, 'cam' + str(cam_num) + 'label_map_T{}.png'.format(str(int(threshold*100))))
        cv2.imwrite(label_map_save_pth,255.0*label2rgb(labels, colors=['red', 'green', 'blue', 'purple', 'pink', 'black']))

        num_good_lesions_found = 0
        regions = regionprops(labels)
        for region in regions:
            roundness = 4*region.area / (np.pi * region.axis_major_length * region.axis_major_length)
            x,y = region.centroid
            if region.area > 500 and roundness > 0.5:
                if x > rows * 0.1 and x < rows * 0.9 and y > cols * 0.1 and y < cols * 0.9:
                    num_good_lesions_found += 1
                else:
                    labels = np.where(labels==region.label,0,labels)
        if num_good_lesions_found == num_lesions:
            best_labels = labels
            break
        elif num_good_lesions_found < min_lesions_over_threshold and num_good_lesions_found >= num_lesions:
            min_lesions_over_threshold = num_good_lesions_found
            enough_lesions_found = True
            best_labels = labels
        elif num_good_lesions_found > max_lesions_under_threshold and enough_lesions_found == False:
            max_lesions_under_threshold = num_good_lesions_found
            best_labels = labels

    labels = best_labels
    counts = []
    # regions doesn't include background
    regions = regionprops(labels)
    for region in regions:
        counts.append([region.area,region.label])
    counts.sort()

    while len(counts) > num_lesions:
        area, label_num = counts.pop(0)
        labels = np.where(labels == label_num, 0, labels)

    # now all the labels will be 1-num_lesion
    labels_unique = np.unique(labels)
    for i,lab_u in enumerate(labels_unique):
        labels = np.where(labels==lab_u,i,labels)

    label_map_ws = watershed(-distance, labels, mask=bin_img)
    label_map_rgb = np.array(255.0*label2rgb(label_map_ws,colors=['red','green','blue','purple','pink','black']),dtype='float32')
    label_map_bgr = cv2.cvtColor(label_map_rgb,cv2.COLOR_RGB2BGR)

    label_map_save_pth = os.path.join(postprocess_dir, 'cam' + str(cam_num) + 'label_map.png')
    cv2.imwrite( label_map_save_pth,label_map_bgr )

    return label_map_ws

def processSegStack(seg_stack,img_stack,num_lesions,labels_ws,imageDF,resize_ratio,postprocess_dir,imgsWLesions_dir):
    seg_stack = seg_stack.copy()
    # add first fake segmentation of zero to make processing easier
    num_imgs,rows,cols = seg_stack.shape
    first_seg = np.zeros((1,rows,cols))
    seg_stack = np.concatenate((first_seg,seg_stack),axis=0)

    alpha = 0.4
    for i in range(num_imgs):
        df_index = i*num_lesions
        img_name = imageDF.loc[df_index,'Image Name']
        leaf_img = np.array(img_stack[i],dtype='float32') # leaf_img is RGB

        original_img = leaf_img.copy()
        original_seg = original_img.copy()
        original_seg[:,:,0] = np.where(seg_stack[i+1,:,:]==1,255,original_seg[:,:,0])
        original_seg[:,:,1] = np.where(seg_stack[i+1,:,:]==1,0,original_seg[:,:,1])
        original_seg[:,:,2] = np.where(seg_stack[i+1,:,:]==1,0,original_seg[:,:,2])
        orig_seg_img = cv2.addWeighted(original_img,1-alpha,original_seg,alpha,0.0)
        orig_seg_img_pth = os.path.join(postprocess_dir, img_name.replace('.png','') + '_OSeg.png')
        cv2.imwrite(orig_seg_img_pth,cv2.cvtColor(orig_seg_img,cv2.COLOR_RGB2BGR))

        # current image is the previous segmentation AND the current segmentation
        # so that lesions don't randomly disappear
        seg_stack[i+1,:,:] = seg_stack[i+1,:,:] + seg_stack[i,:,:]
        continuous_seg = seg_stack[i+1,:,:]
        curr_labels = np.where(continuous_seg>0, labels_ws, 0)
        num_lesions_segmented = np.max(curr_labels)
        seg_image = 255.0*label2rgb(curr_labels,colors=['red','green','blue','purple','pink','black'])

        seg_image_mask = np.sum(seg_image,axis=2,keepdims=True)
        seg_image_mask = np.where(seg_image_mask>0,1,0)

        overlay = np.where(seg_image_mask==1,seg_image,leaf_img)
        overlay = np.array(overlay,dtype='float32')

        img_w_lesions = cv2.addWeighted(leaf_img,1-alpha,overlay,alpha,0.0)
        img_w_lesions_pth = os.path.join(imgsWLesions_dir, img_name.replace('.png','_lesions.png'))
        cv2.imwrite(img_w_lesions_pth,cv2.cvtColor(img_w_lesions,cv2.COLOR_RGB2BGR))

        lesion_total = 0
        for l in range(num_lesions_segmented):
                area_str = 'Lesion #'
                imageDF.at[df_index+l, area_str] = l+1
                lesion_size = np.count_nonzero(np.where(curr_labels==l+1,l+1,0))
                imageDF.at[df_index+l,'Lesion Area Pixels'] = lesion_size
                imageDF.at[df_index+l, 'Adjusted Lesion Pixels'] = lesion_size * resize_ratio
                lesion_total += lesion_size

        lesion_avg = lesion_total / num_lesions_segmented

        # Only record Avg Adj Pixel Size one time per image for easier statistical analysis
        #for l in range(num_lesions):
        #    imageDF.at[df_index+l, 'Avg Adj Pixel Size'] = lesion_avg * resize_ratio
        imageDF.at[df_index, 'Avg Adj Pixel Size'] = lesion_avg * resize_ratio

    return imageDF

def cleanDF(df):
    clean_df = df[[
        'Image Name',
        'File Location',
        'CameraID',
        'Year',
        'Month',
        'Day',
        'Hour',
        'Innoc Year',
        'Innoc Month',
        'Innoc Day',
        'Innoc Hour',
        'Hours Elapsed',
        'Lesion #',
        'Lesion Area Pixels',
        'ResizeRatio',
        'Adjusted Lesion Pixels',
        'Avg Adj Pixel Size',
        'Camera #',
        'Array #',
        'Leaf #',
        'Leaf Section #',
        'Covariable 1',
        'Covariable 2',
        'Covariable 3',
        'Covariable 4',
        'Vector Name',
        'Gene of Interest',
        'Comments',
        'Description']]
    return clean_df