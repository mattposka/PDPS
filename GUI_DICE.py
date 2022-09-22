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
#from utils.datasets import LEAFTest
from utils.datasetsFull import LEAFTest
import postprocess as postp

import model.u_netDICE as u_netDICE
import model.u_netDICE_Brown as u_netDICE_Brown

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

import shutil

from skimage.segmentation import watershed
from skimage import measure
from skimage.feature import peak_local_max
from scipy import ndimage

# There is a UserWarning for one pytorch layer that I dont' want printing
warnings.filterwarnings('ignore',category=UserWarning)

imagelst = []
current = 0
count = 0

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GUI(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        parent.title('Leaf Image Processor')
        parent.minsize(640, 400)

        defaultFont = tkFont.nametofont( 'TkDefaultFont' )
        defaultTextFont = tkFont.nametofont( 'TkTextFont' )
        defaultFont.configure( size=24 )
        defaultTextFont.configure( size=24 )

        self.parent.grid_rowconfigure( 0,weight=3 )
        self.parent.grid_rowconfigure( 1,weight=1 )
        self.parent.grid_columnconfigure( (0,1,2,3),weight=1 )

        self.setImageFrame() 
        self.setModelFrame()
        self.setDirFrame()
        self.setLesionFrame()
        self.setInnocFrame()
        self.setMetaFrame()
        self.setDescFrame()
        self.setRunFrame()

    def setImageFrame( self, ):
        self.imageFrame = tk.Frame( root,bg='#65C1E8',borderwidth=15,relief='ridge' )
        self.nextbutton()
        self.previousbutton()

        self.imageFrame.grid_rowconfigure( (0,1,3),weight=1 )
        self.imageFrame.grid_rowconfigure( (2,),weight=3 )
        self.imageFrame.grid_columnconfigure( (0,1),weight=1 )
        self.imageFrame.grid( row=0,column=0,sticky='nsew' )

        self.selectbutton()

    def setModelFrame( self, ):
        self.modelFrame = tk.Frame( root,bg='#D85B63',borderwidth=15,relief='ridge' )

        self.modelTypeVar = tk.StringVar( self.modelFrame )
        self.modelTypeVar.set( 'LEAF_UNET_DICE_NOV21.pth' )
        models_available_list = self.getModelsAvailable()
        self.modelTypeMenu = tk.OptionMenu( self.modelFrame,self.modelFrame,self.modelTypeVar,*models_available_list )
        self.modelTypeMenu.grid( row=1,column=0 )

        self.modelFrame.grid_rowconfigure( (0,1),weight=1 )
        self.modelFrame.grid_columnconfigure( 0,weight=1 )
        self.modelFrame.grid( row=1,column=0,sticky='nsew' )

    def setDirFrame( self, ):
        self.dirFrame = tk.Frame( root,bg='#D680AD',borderwidth=15,relief='ridge' )
        folderLabel = tk.Label( self.dirFrame,text='Set Save Folder Name :',anchor='center' )
        folderLabel.grid( row=0,column=0,sticky=''  )
        self.saveNameEntry = tk.Entry( self.dirFrame )
        self.saveNameEntry.insert( END,'default' )
        self.saveNameEntry.grid( row=1,column=0,sticky='' )

        separator = Separator( self.dirFrame,orient='horizontal' )
        separator.grid( row=2,sticky='ew' )

        self.dirFrame.grid_rowconfigure( (0,1,2,3,4,5),weight=1 )
        self.dirFrame.grid_columnconfigure( (0),weight=1 )
        self.dirFrame.grid( row=0,column=1,sticky='nsew' )

        self.selectsavedir()

    def setLesionFrame( self, ):
        self.lesionFrame = tk.Frame( root,bg='#5C5C5C',borderwidth=15,relief='ridge' )
        lesionLabel = tk.Label( self.lesionFrame,text='Number of Lesions',anchor='center' )
        lesionLabel.grid( row=0,column=0,sticky=''  )

        self.numberOfLesionsEntry = tk.Entry( self.lesionFrame )
        self.numberOfLesionsEntry.insert( END,'4' )
        self.numberOfLesionsEntry.grid( row=1,column=0,sticky='' )

        self.lesionFrame.grid_rowconfigure( (0,1),weight=1 )
        self.lesionFrame.grid_columnconfigure( 0,weight=1 )
        self.lesionFrame.grid( row=1,column=1,sticky='nsew' )

    def setInnocFrame( self, ):
        self.innocFrame = tk.Frame( root,bg='#C0BA80',borderwidth=15,relief='ridge' )

        self.innocTitle = tk.Label( self.innocFrame,text='Set Innoculation start Time',anchor='center' )
        self.innocTitle.grid( columnspan=2,row=0,sticky=''  )

        self.innocYearLabel = tk.Label( self.innocFrame,text='Year :' )
        self.innocYearLabel.grid( row=1,column=0,sticky='e' )
        self.innocYearTimeVar = tk.Entry( self.innocFrame )
        self.innocYearTimeVar.insert( END,'2000' )
        self.innocYearTimeVar.grid( row=1,column=1,sticky='w' )

        self.innocMonthLabel = tk.Label( self.innocFrame,text='Month :' )
        self.innocMonthLabel.grid( row=2,column=0,sticky='e' )
        self.innocMonthTimeVar = tk.Entry( self.innocFrame )
        self.innocMonthTimeVar.insert( END,'2' )
        self.innocMonthTimeVar.grid( row=2,column=1,sticky='w' )

        self.innocDayLabel = tk.Label( self.innocFrame,text='Day :' )
        self.innocDayLabel.grid( row=3,column=0,sticky='e' )
        self.innocDayTimeVar = tk.Entry( self.innocFrame )
        self.innocDayTimeVar.insert( END,'2' )
        self.innocDayTimeVar.grid( row=3,column=1,sticky='w' )

        self.innocHourLabel = tk.Label( self.innocFrame,text='Hour :' )
        self.innocHourLabel.grid( row=4,column=0,sticky='e' )
        self.innocHourTimeVar = tk.Entry( self.innocFrame )
        self.innocHourTimeVar.insert( END,'2' )
        self.innocHourTimeVar.grid( row=4,column=1,sticky='w' )

        self.innocFrame.grid_rowconfigure( (0,1,2,3,4),weight=1 )
        self.innocFrame.grid_columnconfigure( (0,1),weight=1 )
        self.innocFrame.grid( row=0,rowspan=1,column=2,sticky='nsew' )

    def setMetaFrame( self, ):
            self.metaFrame = tk.Frame( root,bg='#C0BA80',borderwidth=15,relief='ridge' )

            self.metaFrame.grid_rowconfigure( (0,1,2),weight=1 )
            self.metaFrame.grid_columnconfigure( 0,weight=1 )
            self.metaFrame.grid( row=1,column=2,sticky='nsew' )

            self.selectmetafile = tk.Button( self.metaFrame,text='Select Experimental Design File',command=self.setMetaFile )
            self.selectmetafile.grid( row=0,column=0,sticky='' )

            self.saveMetaLabel = tk.Label( self.metaFrame,text='Experimental Design File:' )
            self.saveMetaLabel.grid( row=1,column=0,sticky='s' )

            self.metaFrame.update()
            self.metaFile = 'None'
            self.metaFileVar = tk.StringVar()
            self.metaFileLabel = tk.Label( self.metaFrame,textvariable=self.metaFileVar,wraplength=int(0.9*(self.metaFrame.winfo_width())) )
            self.metaFileLabel.grid( row=3,column=0,sticky='n' )

    def setDescFrame( self, ):
        self.descFrame = tk.Frame( root,bg='#FDC47D',borderwidth=15,relief='ridge' )

        descLabel = tk.Label( self.descFrame,text="Description of this experiment and image series",anchor='center' )
        descLabel.grid( column=1,row=0,sticky='' )

        self.e1 = tk.Text( self.descFrame,height=4,width=20,font=('Helvetica',24) )
        self.e1.grid( column=1,row=1,rowspan=2,sticky='nsew' )

        self.savebutton()

        self.descFrame.grid_rowconfigure( (0,1,2,3),weight=1 )
        self.descFrame.grid_columnconfigure( (0,2),weight=1 )
        self.descFrame.grid_columnconfigure( 1,weight=3 )
        self.descFrame.grid( row=0,column=3,sticky='nsew' )

    def setRunFrame( self, ):
        self.runFrame = tk.Frame( root,bg='#EA3B46',borderwidth=15,relief='ridge' )

        self.gpuNumLabel = tk.Label( self.runFrame,text='Select Which GPU\nOr -1 For CPU' )
        self.gpuNumLabel.grid( row=0,column=0,sticky='e' )
        self.gpuNumEntry = tk.Entry( self.runFrame,justify='center' )
        self.gpuNumEntry.insert( END,'-1' )
        self.gpuNumEntry.grid( row=0,column=1,sticky='w' )

        self.Runbutton = tk.Button( self.runFrame,text='Run',command=self.process )
        self.Runbutton.grid( row=1,columnspan=2 )

        self.runFrame.grid_rowconfigure( (0,1),weight=1 )
        self.runFrame.grid_columnconfigure( (0,1),weight=1 )
        self.runFrame.grid( row=1,column=3,sticky='nsew' )

    # gets models from the pytorch_models directory
    def getModelsAvailable( self, ):
        models_available = glob.glob( 'pytorch_models/*' )
        models_available_list = []
        for m in models_available:
            models_available_list.append( m.split('/')[-1] )

        modelTypeLabel = tk.Label( self.modelFrame,text='Select Model to Use',anchor='center' )
        modelTypeLabel.grid( column=0,row=0,sticky=''  )

        return models_available_list

    # saves description called /rootName/resultFiles/description.txt
    def save(self):
        # make results directory
        root_dir = self.saveNameEntry.get()
        if root_dir == 'default':
            print( '\nEnter a save file name first!\nusing "default" for now.')

        self.save_file_name = root_dir

        saveDir = self.saveDir
        if saveDir != 'None':
            root_dir = os.path.join( saveDir,self.saveNameEntry.get() )

        results_dir = os.path.join( root_dir,'resultFiles' )
        if not os.path.exists( results_dir ):
            os.makedirs( results_dir )

        description_pth = os.path.join( results_dir,'description.txt' )

        description = self.e1.get('1.0', END)
        with open(description_pth, 'a') as file:
            file.write(str(description) + '\n')
        print('\nDescription saved')

    def savebutton( self, ):
        self.savebutton = tk.Button( self.descFrame,text='save',command=self.save,anchor='center' )
        self.savebutton.grid( column=1,row=3 )

    def selectbutton( self, ):
        self.selectbutton = tk.Button( self.imageFrame,text='Select Images', command=self.openimage)
        self.selectbutton.grid( columnspan=2,row=0)

    def selectsavedir( self, ):
        self.selectsavedir = tk.Button( self.dirFrame,text='Set Results Directory',command=self.setSaveDir )
        self.selectsavedir.grid( row=3,column=0,sticky='' )

        self.saveDirLabel0 = tk.Label( self.dirFrame,text='Results Directory:' )
        self.saveDirLabel0.grid( row=4,column=0,sticky='s' )

        self.dirFrame.update()
        self.saveDir = 'None'
        self.saveDirVar = tk.StringVar()
        self.saveDirLabel = tk.Label( self.dirFrame,textvariable=self.saveDirVar,wraplength=int(0.9*(self.dirFrame.winfo_width())) )
        self.saveDirLabel.grid( row=5,column=0,sticky='n' )

    def openimage(self):
        global imagelst
        global current
        current = 0

        self.filenames = filedialog.askopenfilenames(filetypes=( ('all files','*.*'),('png files','*.png'),('jpeg files','*.jpeg') ), initialdir='/', title='Select Image')
        imagelst = self.sortimages( self.filenames )

        self.imageNameLabel = tk.Label( self.imageFrame )
        self.original = Image.open( imagelst[0] )
        self.image = ImageTk.PhotoImage( self.original )
        self.canvas = Canvas( self.imageFrame,bd=0,highlightthickness=0 )
        self.canvasArea = self.canvas.create_image( 0,0,image=self.image,anchor=NW )
        self.canvas.grid( row=2,columnspan=2,sticky=W+E+N+S)
        self.canvas.bind( "<Configure>",self.resize )

        self.imageFrame.update()
        self.imageNameVar = tk.StringVar()
        self.imageNameVar.set( imagelst[ current ] )
        self.imageNameLabel = tk.Label( self.imageFrame,textvariable=self.imageNameVar,wraplength=int(0.9*(self.canvas.winfo_width())) )
        self.imageNameLabel.grid( row=1,columnspan=2,sticky='s' )

    def resize( self,event ):
        size = (event.width, event.height)
        resized = self.original.resize( size,Image.ANTIALIAS )
        self.image = ImageTk.PhotoImage( resized )
        self.canvas.create_image( 0,0,image=self.image,anchor=NW )
        self.imageNameLabel.config( wraplength=event.width )
        self.imageFrame.update()

    def move(self, delta):
        global current

        current += delta
        if current < 0:
            current = len(imagelst)-1
        if current >= len(imagelst):
            current = 0
        self.imageNameVar.set( imagelst[ current ] )

        self.original = Image.open( imagelst[current] )
        resized = self.original.resize( (self.canvas.winfo_width(),self.canvas.winfo_height()),Image.ANTIALIAS )
        self.image = ImageTk.PhotoImage( resized )
        self.canvas.itemconfig( self.canvasArea,image=self.image )
        self.imageFrame.update()

    def nextbutton(self):
        self.nextbutton = tk.Button( self.imageFrame,text='Next',command=lambda: self.move(+1) )
        self.nextbutton.grid( row=3,column=1 )

    def previousbutton(self):
        self.previousbutton = tk.Button( self.imageFrame,text='Previous',command=lambda: self.move(-1) )
        self.previousbutton.grid( row=3,column=0 )

    def setSaveDir(self):
        self.saveDir = filedialog.askdirectory( initialdir='/',title='Select Save Directory' )
        self.saveDirVar.set( self.saveDir )

    def setMetaFile(self):
        self.metaFile = filedialog.askopenfilename( filetypes=( ('excel files','*.xlsx'),('csv files','*.csv'),('all files','*.*') ), initialdir='/', title='Select Experimental Design File')
        self.metaFileVar.set( self.metaFile )

    # sorts images by time and constructs a dataframe to hold them
    def sortimages( self,filenames ):
        imagemat = []
        for i in filenames:
            imname = i.split('/')[-1]
            cameraID = int( imname[9:11] )
            year = int( imname[12:16] )
            month = int( imname[17:19] )
            day = int( imname[20:22] )
            hour = int( int( imname[23:27] ) / 100 )
            #TODO make this self.num_lesions later
            for j in range(4):
                imagemat.append([imname, i, cameraID, year, month, day, hour])

        self.imageDF = pd.DataFrame( imagemat,columns=['Image Name','File Location','CameraID','Year','Month','Day','Hour'] )
        self.imageDF['Lesion #'] = ''
        self.imageDF['Lesion Area Pixels'] = ''
        self.imageDF['Adjusted Lesion Pixels'] = ''
        self.imageDF['Camera #'] = self.imageDF['CameraID']
        self.imageDF = self.imageDF.sort_values( by=['CameraID','Year','Month','Day','Hour'] )
        self.imageDF['index_num'] = np.arange( len(self.imageDF) )
        self.imageDF = self.imageDF.reset_index(drop=True)
        imagelst = self.imageDF['File Location']
        return imagelst

    # Formats the original dataframe based off of the images selected
    def formatDF( self, ):
        description = self.e1.get('1.0', END)
        self.imageDF['Avg Adj Pixel Size'] = ''
        self.imageDF['Description'] = description

        innoc_year = int( self.innocYearTimeVar.get() )
        innoc_month = int( self.innocMonthTimeVar.get() )
        innoc_day = int( self.innocDayTimeVar.get() )
        innoc_hour = int( self.innocHourTimeVar.get() )
        if innoc_year==2000 and innoc_month==2 and innoc_day==2 and innoc_hour==2:
            innoc_year = self.imageDF.loc[0,'Year']
            innoc_month = self.imageDF.loc[0,'Month']
            innoc_day = self.imageDF.loc[0,'Day']
            innoc_hour = self.imageDF.loc[0,'Hour']

        self.imageDF[ 'Innoc Year' ] = innoc_year
        self.imageDF[ 'Innoc Month' ] = innoc_month
        self.imageDF[ 'Innoc Day' ] = innoc_day
        self.imageDF[ 'Innoc Hour' ] = innoc_hour
        hours_elapsed = []
        start_datetime = dt.datetime( innoc_year,
                        innoc_month,
                        innoc_day,
                        innoc_hour
                        ) 
        for df_index,df_row in self.imageDF.iterrows():
            end_datetime = dt.datetime( df_row['Year'],df_row['Month'],df_row['Day'],df_row['Hour'] )
            time_diff = end_datetime - start_datetime
            secs_diff = time_diff.total_seconds()
            hours_diff = np.divide( secs_diff,3600 )
            hours_elapsed.append( int( hours_diff ) )
        self.imageDF[ 'Hours Elapsed' ] = hours_elapsed
        self.imageDF[ 'ResizeRatio' ] = 0

    def readMetaFile( self, ):
        mFile_name = str(self.metaFile)
        mFile_Found = False
        if mFile_name != '' and mFile_name != 'None':
            mFile = pd.read_excel(mFile_name,engine='openpyxl')
            mFile_Found = True


        # These are all of the columns in the meta file
        self.imageDF[ 'Array #' ] = ''
        self.imageDF[ 'Leaf #' ] = ''
        self.imageDF[ 'Leaf Section #' ] = ''
        self.imageDF[ 'Covariable 1' ] = ''
        self.imageDF[ 'Covariable 2' ] = ''
        self.imageDF[ 'Covariable 3' ] = ''
        self.imageDF[ 'Covariable 4' ] = ''
        self.imageDF[ 'Vector Name' ] = ''
        self.imageDF[ 'Gene of Interest' ] = ''
        self.imageDF[ 'Comments' ] = ''

        if mFile_Found == True:
            for tuple in self.imageDF.itertuples():
                df_row = tuple[0]
                camera_num = self.imageDF.at[df_row,'CameraID']
                mFile_row = mFile.loc[mFile['Camera #'] == camera_num]
                mFile_row_ri = mFile_row.reset_index(drop=True)

                a = mFile_row_ri.at[0,'Array #']
                self.imageDF.at[df_row,'Array #'] = mFile_row_ri.at[0,'Array #']
                self.imageDF.at[df_row, 'Leaf #'] = mFile_row_ri.at[0, 'Leaf #']
                self.imageDF.at[df_row,'Leaf Section #' ] = mFile_row_ri.at[0,'Leaf Section #']
                self.imageDF.at[df_row,'Covariable 1' ] = mFile_row_ri.at[0,'Covariable 1']
                self.imageDF.at[df_row,'Covariable 2' ] = mFile_row_ri.at[0,'Covariable 2']
                self.imageDF.at[df_row,'Covariable 3' ] = mFile_row_ri.at[0,'Covariable 3']
                self.imageDF.at[df_row,'Covariable 4' ] = mFile_row_ri.at[0,'Covariable 4']
                self.imageDF.at[df_row,'Vector Name' ] = mFile_row_ri.at[0,'Vector Name']
                self.imageDF.at[df_row,'Gene of Interest' ] = mFile_row_ri.at[0,'Gene of Interest']
                self.imageDF.at[df_row,'Comments' ] = mFile_row_ri.at[0,'Comments']

    #TODO Test what n does
    def process(self,n=''):

        n = int( self.gpuNumEntry.get() )
        if n != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = n

        self.num_lesions = self.numberOfLesionsEntry.get()
        num_lesions = self.num_lesions
        print( '\nUsing {} lesions!'.format(num_lesions) )
        self.num_lesions = int( num_lesions )

        root_dir = self.saveNameEntry.get()
        if root_dir == 'default':
            print( '\nEnter a save file name first!\nusing "default" for now.')

        # save description text
        self.save()

        saveDir = self.saveDir
        if saveDir != 'None':
            root_dir = os.path.join( saveDir,self.saveNameEntry.get() )

        self.model_name = self.modelTypeVar.get()

        result_dir = os.path.join( root_dir,'resultFiles' )
        if not os.path.exists( result_dir ):
            os.makedirs( result_dir )
        result_file = os.path.join(result_dir, self.save_file_name+'.csv')

        postprocess_dir = os.path.join( root_dir,'postprocessing' )
        if not os.path.exists( postprocess_dir ):
            os.makedirs( postprocess_dir )

        txt_dir = os.path.join( root_dir,'txt' )
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        imgsWLesions_dir = os.path.join( result_dir,'imgsWLesions' )
        if not os.path.exists(imgsWLesions_dir):
            os.makedirs(imgsWLesions_dir)

        model_dir = os.path.join( 'pytorch_models' )
        if not os.path.exists( model_dir ):
            os.makedirs( model_dir )

        # choose pytorch model based off of model type
        model_id = self.model_name
        model_pth = os.path.join( model_dir,( model_id ) )
        RESTORE_FROM = model_pth
        print("\nModel restored from:", RESTORE_FROM)
        ###################################################################
        
        self.formatDF()

        print('self.metaFile :',self.metaFile)
        self.readMetaFile()

        #TODO maybe do all of the model-specific stuff here
        modelname = self.modelTypeVar.get()
        patch_size = 512

        IMG_MEAN = np.array((128.95671, 109.307915, 96.25992), dtype=np.float32) # R G B

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print( '\tUsing device:',device )
        if device.type == 'cuda':
            print( '\t',pytorch.cuda.get_device_name(0) )

        #TODO make this cleaner later, but for now just load different model if it has the specific name
        NUM_CLASSES = 2
        model = u_netDICE_Erode_Run.UNetDICE_Erode_Run(NUM_CLASSES)
        if model_id == 'LEAF_UNET_DICE_NOV21.pth':
            model = u_netDICE.UNetDICE(NUM_CLASSES)

        model = nn.DataParallel(model)
        model.to(device)
        saved_state_dict = torch.load(RESTORE_FROM, map_location=lambda storage, loc: storage)

        model.load_state_dict(saved_state_dict['state_dict'])
        model.eval()

        num_unique_leaves = len(np.unique(self.imageDF['CameraID']))
        leaf_seg_stack = []
        leaf_img_stack = []
        start_image_df_idx = None
        leafMask = None
        row_mid = 0
        col_mid = 0
        half_side = 0
        resize_ratio = 0
        cams_completed = 1
        for df_index,df_row in self.imageDF.iterrows():
            if df_index % self.num_lesions != 0:
                continue

            test_img_pth = df_row[ 'File Location' ]
            filename = test_img_pth.split('/')[-1] # name of the image

            # image_fg_size is the size of the side of the square containing the entire unresized leaf
            # save this number for later use when to determine the actual size of all of the lesions later

            new_leaf = True
            prev_img_df = self.imageDF[
                (self.imageDF['CameraID'] == self.imageDF.loc[df_index, 'CameraID']) &
                (self.imageDF['Year'] <= self.imageDF.loc[df_index, 'Year']) &
                (self.imageDF['Month'] <= self.imageDF.loc[df_index, 'Month']) &
                (self.imageDF['Day'] <= self.imageDF.loc[df_index, 'Day']) &
                (self.imageDF['index_num'] < df_index)
                ]
            if len(prev_img_df) > 0:
                new_leaf = False

            if new_leaf == True:
                if leaf_seg_stack:
                    print('\nPostProcessing now!')
                    sum_stack,label_map_ws = postp.watershedSegStack(np.array(leaf_seg_stack),self.num_lesions,postprocess_dir,self.imageDF,start_image_df_idx)
                    self.imageDF = postp.processSegStack(np.array(leaf_seg_stack),leaf_img_stack,self.num_lesions,label_map_ws,self.imageDF,start_image_df_idx,resize_ratio,postprocess_dir,imgsWLesions_dir)
                    cams_completed += 1

                print('\nProcessing {}/{} Cameras'.format(cams_completed, num_unique_leaves))
                print('Processing image :', filename)
                start_image_df_idx = df_index
                resized_image, normalized_image, leaf_mask, resize_ratio, half_side, row_mid, col_mid \
                    = process_tif(test_img_pth)
                leaf_img_stack = [resized_image]
                leaf_seg_stack = []
                leafMask = leaf_mask
            else:
                resized_image, normalized_image, resize_ratio, = quick_process_tif(test_img_pth,leafMask,row_mid,col_mid,half_side )
                leaf_img_stack.append(resized_image)
                print( 'Processing image :',filename )

            for k in range(self.num_lesions):
                self.imageDF.at[ df_index+k,'ResizeRatio' ] = resize_ratio

            # Add 4th channel (blank or prev_segmentation)
            h,w,c = normalized_image.shape
            input_image = np.zeros( (h,w,4) )
            input_image[:,:,:3] = normalized_image

            with torch.no_grad():
                formatted_img = np.transpose(input_image,(2,0,1)) # transpose because channels first

                formatted_img = formatted_img.astype(np.float32)
                image_tensor = torch.from_numpy(np.expand_dims(formatted_img,axis=0))
                output = model(image_tensor).to(device)

                msk = torch.squeeze(output).data.cpu().numpy()
                msk = np.where(msk>0,1,0)

            leaf_seg_stack.append(msk)
#####################################################################################################################
#####################################################################################################################

        if leaf_seg_stack:
            print('\nPostProcessing now!')
            sum_stack, label_map_ws = postp.watershedSegStack(np.array(leaf_seg_stack), self.num_lesions,
                                                              postprocess_dir, self.imageDF, start_image_df_idx)
            self.imageDF = postp.processSegStack(np.array(leaf_seg_stack), leaf_img_stack, self.num_lesions, label_map_ws,
                                  self.imageDF, start_image_df_idx, resize_ratio, postprocess_dir, imgsWLesions_dir)

        clean_df = self.imageDF.copy()

        # TODO Add Lesion # Column
        reformatted_csv_df = clean_df[[
        #reformatted_csv_df = reformatted_csv_df[[
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

        reformatted_csv.to_csv( result_file,mode='a',header=not os.path.exists(result_file,index=False))

        print( '\n\tresult_file located :',result_file )
        print( '\n****************************************************************************' )
        print( '***********************************DONE*************************************' )
        print( '****************************************************************************' )

if __name__ == "__main__":
    root = tk.Tk()
    GUI(root)
    root.mainloop()
