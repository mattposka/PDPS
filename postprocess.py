import numpy as np
import pandas as pd

from merge_npz_finalFull import merge_npz
from scipy.sparse import load_npz, save_npz, csr_matrix, coo_matrix
from scipy.misc import imread, imsave
import scipy.misc as spm
from PIL import Image
from utils.transforms import vl2im, im2vl
from skimage import filters
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects
import cv2

# Not used right now
# keeps the largest [num_lesions] lesions
def numLesionsFilter( new_labeled_img,num_lesions ):
    label_props_area_list = np.zeros( shape=[len(new_labeled_img),2] )
    label_props = regionprops( new_labeled_img )  
    for l,prop in enumerate(label_props):
        label_props_area_list[ l,0 ] = prop.label
        label_props_area_list[ l,1 ] = prop.area
    sorted_area_list = label_props_area_list[ np.argsort( label_props_area_list[:,1] ) ]
    areas_to_remove = sorted_area_list[ :(-1*num_lesions),: ]

    for area in range( len( areas_to_remove ) ):
        new_labeled_img = np.where( new_labeled_img==areas_to_remove[area,0],0,new_labeled_img )

    return new_labeled_img

# fills doughnuts, but objects must be fully closed for it to work
def fillHoles( img ):
    ##################################################################################################
    # floolfill stuf
    ##################################################################################################
    img_copy = img.copy()
    img_copy[ img_copy!=0 ] = 255
    holes = img_copy.copy()
    h,w = img.shape[:2]
    mask_to_fill = np.zeros( (h+2,w+2),np.uint8 )
    # Floodfill from point (0,0) (known background)
    cv2.floodFill( holes,mask_to_fill,(0,0),255 )
    # Invert floodfilled image
    holes = cv2.bitwise_not( holes )
    filled_holes = cv2.bitwise_or( img_copy,holes )
    ##################################################################################################
    filled_holes[ filled_holes!=0 ] = 1
    img = filled_holes
    return img

# This will try to combine regions that are very close to one another
#TODO check expand_ratio - changed to zero to fix very long lesions
def combineRegions( labeled_img,ref_ecc,pred_img_pth,leaf_mask,expand_ratio=0,mal=450 ):
    circle_img = np.asarray( labeled_img,dtype=np.uint8 )
    contours,heir = cv2.findContours( circle_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )

    centers = []
    for c in contours:
        (x,y),r = cv2.minEnclosingCircle( c ) 
        if r > 5 and r < int(mal/2): # mal is maximum radius
            centers.append( [x,y,r] )

    # draws a circle over every segmented region that is slightly larger than the region
    for c in centers:
       cv2.circle( circle_img,(int(c[0]),int(c[1])),int(expand_ratio*c[2]),(255),-1 )
    #cir_img_pth = pred_img_pth.replace( '.png','_circles.png' )
    #imsave( cir_img_pth,circle_img )

    # labeled the new circle image
    labeled_circle = label(circle_img, connectivity=2)
    #lab_cir_pth = pred_img_pth.replace( '.png','_labeledCircles.png' )
    #imsave( lab_cir_pth,labeled_circle )

    # applies the labels of the circle image to the original image,
    # theoretically grouping regions close to one-another together.
    labeled_circle[ labeled_img==0 ] = 0
    labeled_image = labeled_circle
    #lab_img_pth = pred_img_pth.replace( '.png','_labeledRegions.png' )
    #imsave( lab_img_pth,labeled_image )

    # remove small regions
    new_props = regionprops(labeled_image)
    labeled_img = regionAreaFilter( new_props,labeled_image )
    #lab_img_pth = pred_img_pth.replace( '.png','_RegionAreaFilter.png' )
    #imsave( lab_img_pth,labeled_image )

    # remove non-circle region
    new_props = regionprops(labeled_image)
    labeled_img = circleFilter( new_props,labeled_image,ref_ecc=ref_ecc )
    #lab_img_pth = pred_img_pth.replace( '.png','_NonCircleFilter.png' )
    #imsave( lab_img_pth,labeled_image )

    #TODO test leaf_mask filter again later
    #new_props = regionprops(labeled_image)
    #labeled_img = leafMaskFilter( new_props,labeled_image,leaf_mask )
    #lab_img_pth = pred_img_pth.replace( '.png','_LeafMaskFilter.png' )
    #imsave( lab_img_pth,labeled_image )

    return labeled_img

# remove small regions
def regionAreaFilter( new_props,labeled_img,min_lesion_area=35 ):
    for i, reg in enumerate(new_props):
        if reg.area < min_lesion_area:
            labeled_img[labeled_img == reg.label] = 0
    return labeled_img

# remove non-circle region
def circleFilter( new_props,labeled_img,ref_ecc ):
    for i, reg in enumerate(new_props):
        if reg.eccentricity > ref_ecc:
            labeled_img[labeled_img == reg.label] = 0
        #print('reg.equivalent_diameter :',reg.equivalent_diameter )
        #print('reg.major_axis_length :',reg.major_axis_length )
        if reg.equivalent_diameter < 0.6*reg.major_axis_length: 
            labeled_img[labeled_img == reg.label] = 0
    return labeled_img

# remove lesion areas that are outside of the leaf mask
def leafMaskFilter( new_props,labeled_img,leaf_mask ):
    for i, reg in enumerate(new_props):
        bb = reg.bbox #min_row,min_col,max_row,max_col
        # [min,max) -> have to -1 to the max
        corners = [[bb[0],bb[1]],
                        [bb[0],bb[3]-1],
                        [bb[2]-1,bb[1]],
                        [bb[2]-1,bb[3]-1],
                        ]
        for c in corners:
            if not leaf_mask[c[0],c[1]]:
                labeled_img[labeled_img == reg.label] = 0
    return labeled_img

# draws cirles inside of every region to fill holes, doughnuts, and crescents
def drawCircles( labeled_img,postProcessed_img_pth ):

    new_labeled_img8 = np.asarray( labeled_img,dtype=np.uint8 )
    contours,heir = cv2.findContours( new_labeled_img8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )

    centers = []
    for c in contours:
        (x,y),r = cv2.minEnclosingCircle( c ) 
        if r > 5:
            centers.append( [x,y,r] )

    new_labeled_img_3d = cv2.cvtColor( new_labeled_img8,cv2.COLOR_GRAY2BGR )
    for c in centers:
       cv2.circle( new_labeled_img_3d,(int(c[0]),int(c[1])),int(0.7*c[2]),(0,255,0),2 )
    #circleimgpath = postProcessed_img_pth.replace( '.png','_circleFill.png' )
    #imsave( circleimgpath,new_labeled_img_3d )

    new_img = cv2.cvtColor( new_labeled_img_3d,cv2.COLOR_BGR2GRAY )
    new_img[ new_img != 0 ] = 1
    new_img = fillHoles( new_img )

    return new_img

# Sort by contour size and take n_lesion largest areas, then sort by x+y locations
def sortAndFilterContours( contour_arr,imgsWLesions_dir,df_index,num_lesions ):
    # This is where the largest [num_lesions] lesions are kept
    # also sorts by y and x values
    #contour_arr = contour_arr[ contour_arr[:,7].argsort() ][-num_lesions:,:]
    contour_arr = contour_arr[contour_arr[:, 7].argsort()][:-1,:] #remove the biggest region because it iskground
    contour_arr = contour_arr[contour_arr[:,7].argsort()][-num_lesions:,:]
    contour_arr = contour_arr[ contour_arr[:,5].argsort() ]
    contour_arr = contour_arr[ contour_arr[:,6].argsort(kind='mergesort') ]

    return contour_arr

# Checks the lesions order to the previous lesions in the same leaf
# TODO probably need to fix this
def checkLesionOrder( imageDF,df_index,contours_ordered,num_lesions ):
    prev_img_df = imageDF[ 
                    (imageDF['cameraID']==imageDF.loc[df_index,'cameraID']) &
                    (imageDF['year']==imageDF.loc[df_index,'year']) &
                    (imageDF['month']==imageDF.loc[df_index,'month']) &
                    (imageDF['day']==imageDF.loc[df_index,'day']) &
                    #(imageDF['index_num']<df_index) &
                    (imageDF['index_num'] < df_index)
                    #TODO maybe we still want to only compare against prev good imgs?
                    #(imageDF['index_num'].isin(good_df_indices))
                    #(imageDF['index_num'].isin(good_df_indices))
                    ]
    # Here contours_ordered will be:
    # [ w*h,x,y,x+w,y+h,cx,cy,area ]
    if len(prev_img_df) > 0:
        prev_img_df = prev_img_df.reset_index(drop=True)
        pd.set_option('display.max_columns', None)
        dfl = len( prev_img_df )
        contours_reordered = np.zeros( shape=(num_lesions,8) )
        lesion_number_taken = []
        for i in range( num_lesions ):
            xs = prev_img_df.at[dfl-1,'l'+str(i+1)+'_xstart']
            xe = prev_img_df.at[dfl-1,'l'+str(i+1)+'_xend']
            ys = prev_img_df.at[dfl-1,'l'+str(i+1)+'_ystart']
            ye = prev_img_df.at[dfl-1,'l'+str(i+1)+'_yend']

            found = False
            if xs != 0 and xe != 0 and ys != 0 and ye != 0:
                for j in range( len(contours_ordered) ):
                    cx = contours_ordered[j,5]
                    cy = contours_ordered[j,6]
                    if cx > xs and cx < xe and cy > ys and cy < ye and found == False:
                        contours_reordered[i,:] = contours_ordered[j,:]
                        found = True
                        lesion_number_taken.append(j)
            else:
                for j in range( len(contours_ordered) ):
                    if j not in lesion_number_taken and found == False:
                        contours_reordered[i,:] = contours_ordered[j,:]
                        lesion_number_taken.append(j)
                        found = True
    else:
        contours_reordered = contours_ordered
    return contours_reordered

# Adds reordered contours to the DF
def addContoursToDF( imageDF,contours_reordered,df_index,num_lesions ):
    for l in range( num_lesions ):
        if l < len(contours_reordered):
            area_str = 'l'+str(l+1)+'_area'
            imageDF.at[ df_index,area_str ] = contours_reordered[l,7]
            imageDF.at[ df_index,'l'+str(l+1)+'_centerX' ] = contours_reordered[l,5]
            imageDF.at[ df_index,'l'+str(l+1)+'_centerY' ] = contours_reordered[l,6]
            imageDF.at[ df_index,'l'+str(l+1)+'_xstart' ] = contours_reordered[l,1]
            imageDF.at[ df_index,'l'+str(l+1)+'_xend' ] = contours_reordered[l,3]
            imageDF.at[ df_index,'l'+str(l+1)+'_ystart' ] = contours_reordered[l,2]
            imageDF.at[ df_index,'l'+str(l+1)+'_yend' ] = contours_reordered[l,4]
        else:
            area_str = 'l'+str(l+1)+'_area'
            imageDF.at[ df_index,area_str ] = 0
            imageDF.at[ df_index,'l'+str(l+1)+'_centerX' ] = 0
            imageDF.at[ df_index,'l'+str(l+1)+'_centerY' ] = 0
            imageDF.at[ df_index,'l'+str(l+1)+'_xstart' ] = 0
            imageDF.at[ df_index,'l'+str(l+1)+'_xend' ] = 0
            imageDF.at[ df_index,'l'+str(l+1)+'_ystart' ] = 0
            imageDF.at[ df_index,'l'+str(l+1)+'_yend' ] = 0

    #slf_size = 1000
    #slf = contours_reordered[:num_lesions,7]
    #too_small = False
    #for i in slf:
    #    if i < 1000:
    #        too_small = True

    return imageDF

# check if segmentation was good
def checkSegQuality( contours_reordered ):
    areas = contours_reordered[:,7]

    goodSeg = False
    if len(areas) > 0:
        mean_area = np.mean(areas)
        max_area = np.max(areas)
        min_area = np.min(areas)

        goodSeg = True
        if min_area < 0.5*mean_area:
            goodSeg = False
        if max_area > 0.5*mean_area:
            goodSeg = False

    return goodSeg

# not using right now
##################################################################################################
##TODO
##if len( contours_reordered ) < self.num_lesions or 0 in contours_reordered or too_small == True:
#if len(contours_reordered) < self.num_lesions or 0 in contours_reordered:
#    self.bad_df_indices.append( df_index )
#else:
#    self.good_df_indices.append( df_index )
##################################################################################################