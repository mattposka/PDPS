import cv2
from tkinter import *
import tkinter as tk
from tkinter.ttk import *
import tkinter.font as tkFont
from tkinter import filedialog

from PIL import ImageTk
from operator import itemgetter
import pandas as pd
from openpyxl import load_workbook
import os

from preprocess import process_tif, quick_process_tif
import numpy as np
import torch
from torch.utils import data
import postprocess as postp

import model.u_netDICE as u_netDICE
import model.u_netDICE_Brown as u_netDICE_Brown
#import model.u_netDICE_BrownCE as u_netDICE_BrownCE
import model.u_net2 as u_net

import torch.nn as nn
import imageio as io
from PIL import Image
from utils.transforms import vl2im, im2vl
from skimage import filters
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects
import datetime as dt
import glob
import warnings
import pickle as p

# Formats the original dataframe based off of the images selected
def formatDF( img_dir ):

    filenames = glob.glob( img_dir + '/*')

    imagemat = []
    for i in filenames:
        imname = i.split('/')[-1]
        cameraID = int( imname[9:11] )
        year = int( imname[12:16] )
        month = int( imname[17:19] )
        day = int( imname[20:22] )
        hour = int( int( imname[23:27] ) / 100 )
        #TODO set to num_lesions
        for _ in range(4):
            imagemat.append([imname, i, cameraID, year, month, day, hour])

    imageDF = pd.DataFrame( imagemat,columns=['Image Name','File Location','CameraID','Year','Month','Day','Hour'] )
    imageDF['Lesion #'] = ''
    imageDF['Lesion Area Pixels'] = ''
    imageDF['Adjusted Lesion Pixels'] = ''
    imageDF['Camera #'] = imageDF['CameraID']
    imageDF = imageDF.sort_values( by=['CameraID','Year','Month','Day','Hour'] )
    imageDF['image_id'] = np.arange( len(imageDF) )
    imageDF = imageDF.reset_index(drop=True)

    imageDF['Avg Adj Pixel Size'] = ''

    innoc_year = imageDF.loc[0,'Year']
    innoc_month = imageDF.loc[0,'Month']
    innoc_day = imageDF.loc[0,'Day']
    innoc_hour = imageDF.loc[0,'Hour']

    imageDF[ 'Innoc Year' ] = innoc_year
    imageDF[ 'Innoc Month' ] = innoc_month
    imageDF[ 'Innoc Day' ] = innoc_day
    imageDF[ 'Innoc Hour' ] = innoc_hour
    hours_elapsed = []
    start_datetime = dt.datetime( innoc_year,
                    innoc_month,
                    innoc_day,
                    innoc_hour
                    ) 
    for df_index,df_row in imageDF.iterrows():
        end_datetime = dt.datetime( df_row['Year'],df_row['Month'],df_row['Day'],df_row['Hour'] )
        time_diff = end_datetime - start_datetime
        secs_diff = time_diff.total_seconds()
        hours_diff = np.divide( secs_diff,3600 )
        hours_elapsed.append( int( hours_diff ) )
    imageDF[ 'Hours Elapsed' ] = hours_elapsed
    imageDF[ 'ResizeRatio' ] = 0

    imageDF[ 'Description' ] = ''
    imageDF[ 'Array #' ] = ''
    imageDF[ 'Leaf #' ] = ''
    imageDF[ 'Leaf Section #' ] = ''
    imageDF[ 'Covariable 1' ] = ''
    imageDF[ 'Covariable 2' ] = ''
    imageDF[ 'Covariable 3' ] = ''
    imageDF[ 'Covariable 4' ] = ''
    imageDF[ 'Vector Name' ] = ''
    imageDF[ 'Gene of Interest' ] = ''
    imageDF[ 'Comments' ] = ''

    return imageDF

#TODO Test what n does
def process(model_name,img_dir,model_dir,gpu=0):

    model_name = model_name
    save_file_name = model_name + '2'

    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    num_lesions = 4

    result_dir = 'test_dir_' + model_name
    if not os.path.exists( result_dir ):
        os.makedirs( result_dir )
    result_file = os.path.join(result_dir, save_file_name+'.csv')

    output_dir = os.path.join( result_dir,'resultFiles' )
    if not os.path.exists( output_dir ):
        os.makedirs( output_dir )
    result_file = os.path.join(output_dir, save_file_name+'.csv')

    postprocess_dir = os.path.join( result_dir,'postprocessing' )
    if not os.path.exists( postprocess_dir ):
        os.makedirs( postprocess_dir )

    imgsWLesions_dir = os.path.join( output_dir,'imgsWLesions' )
    if not os.path.exists(imgsWLesions_dir):
        os.makedirs(imgsWLesions_dir)

    imageDF = formatDF( img_dir )
    p.dump(imageDF,open('imageDF.p','wb'))

    # choose pytorch model based off of model type
    NUM_CLASSES = 2
    #model = u_netDICE_Brown.UNetDICE(NUM_CLASSES)
    #model = u_netDICE_BrownCE.UNetDICE_CE(NUM_CLASSES)
    model = u_net.UNet(NUM_CLASSES)
    model = nn.DataParallel(model)
    model.to(device)
    #saved_state_dict = torch.load(os.path.join(model_dir,model_name+'.pth'), map_location=lambda storage, loc: storage)
    saved_state_dict = torch.load(os.path.join(model_dir,model_name+'.pth'))
    print("\nModel restored from : ",os.path.join(model_dir,model_name+'.pth'))

    model.load_state_dict(saved_state_dict)
    model.eval()

    camera_ids = np.unique(imageDF['CameraID'])
    for cams_completed,camera_id in enumerate(camera_ids):

        cameraDF = imageDF.loc[lambda df: df['CameraID']==camera_id, : ]
        new_camera = True

        leaf_seg_stack = []
        leaf_img_stack = []
        leaf_mask = None
        row_mid = 0
        col_mid = 0
        half_side = 0
        resize_ratio = 0
        cams_completed = 1

        for df_index,df_row in cameraDF.iterrows():
            if df_index % num_lesions != 0:
                continue

            test_img_pth = df_row[ 'File Location' ]
            filename = test_img_pth.split('/')[-1] # name of the image

            if new_camera == True:
                print('\nProcessing {}/{} Cameras'.format(cams_completed+1, len(camera_ids)))
                print('Processing image :', filename)
                resized_image, normalized_image, leaf_mask, resize_ratio, half_side, row_mid, col_mid = process_tif(test_img_pth)
                new_camera = False
            else:
                resized_image, normalized_image, resize_ratio, = quick_process_tif(test_img_pth,leaf_mask,row_mid,col_mid,half_side )
                print( 'Processing image :',filename )

            with torch.no_grad():
                formatted_img = np.transpose(normalized_image,(2,0,1)) # transpose because channels first

                formatted_img = formatted_img.astype(np.float32)
                image_tensor = torch.from_numpy(np.expand_dims(formatted_img,axis=0)).to(device)
                output = model(image_tensor).to(device)
                #print('output.shape 0 : ',output.shape)
                #print('torch.max(output) 0 :',torch.max(output))
                #print('torch.min(output) 0 :',torch.min(output))
                if len(output.shape) > 3:
                    #output = torch.softmax(output, axis=1)
                    #print('output.shape 1 : ',output.shape)
                    #print('torch.max(output) 1 :',torch.max(output))
                    #print('torch.min(output) 1 :',torch.min(output))
                    output = torch.argmax(output, axis=1)
                    #print('output.shape 2 : ',output.shape)
                    #print('torch.max(output) 2 :',torch.max(output))
                    #print('torch.min(output) 2 :',torch.min(output))

                msk = torch.squeeze(output).data.cpu().numpy()
                #print('msk.shape :',msk.shape)
                #print('np.max(msk) :',np.max(msk))
                #print('np.min(msk) :',np.min(msk))

            leaf_seg_stack.append(msk)
            leaf_img_stack.append(resized_image)
            for k in range(num_lesions):
                cameraDF.at[ df_index+k,'ResizeRatio' ] = resize_ratio

        print('\nPostProcessing now!')
        label_map_ws = postp.watershedSegStack(np.array(leaf_seg_stack),num_lesions,postprocess_dir,camera_id)
        cameraDF = postp.processSegStack(np.array(leaf_seg_stack),leaf_img_stack,num_lesions,label_map_ws,cameraDF,resize_ratio,postprocess_dir,imgsWLesions_dir)

        clean_df = postp.cleanDF(cameraDF)
        head = False if os.path.exists(result_file) else True
        clean_df.to_csv( result_file,mode='a',header=head ,index=False)

    print( '\n\tresult_file located :',result_file )
    print( '\n****************************************************************************' )
    print( '***********************************DONE*************************************' )
    print( '****************************************************************************' )

if __name__ == "__main__":
    process('IoU',:w
            
            img_dir='./test_imgs',
            model_dir='./models_to_test',
            gpu=0
            )