import cv2
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
from operator import itemgetter
import pandas as pd
from openpyxl import load_workbook
import os
from leaf_divide_test import process_tif
import numpy as np
import time
import torch
from torch.utils import data
from utils.datasets import LEAFTest
from model.u_net import UNet
import torch.nn as nn
from merge_npz_final import merge_npz
from scipy.sparse import load_npz, save_npz, csr_matrix, coo_matrix
from scipy.misc import imread, imsave
import scipy.misc as spm
from PIL import Image
from utils.transforms import vl2im, im2vl
from skimage import filters
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects
from utils.postprocessing import CRFs
import scipy.stats as ss
import matplotlib.pyplot as plt
import pickle as p
import datetime as dt
import glob
Image.MAX_IMAGE_PIXELS = 933120000
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

# remove background
def rmbackground(slide, npz_dir_whole, npz_dir_whole_post, ref_extent, ref_area):
    print( 'slide :',slide )
    print( 'npz_dir_whole :',npz_dir_whole )
    print( 'npz_dir_whole_post :',npz_dir_whole_post )

    low_dim_img = Image.open(slide)
    # HSV - Hue[0,360],Saturation[0,100],Value[0,100]
    low_hsv_img = low_dim_img.convert('HSV')
    #print( 'low_hsv_img.shape :',low_hsv_img.shape )
    _, low_s, _ = low_hsv_img.split()


    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    #img = cv2.imread(slide)
    #
    ## HSV is Hue[0,179], Saturation[0,255], Value[0,255]
    #hsv_img = cv2.cvtColor( img,cv2.COLOR_BGR2HSV )
    #hue = hsv_img[:,:,0]
    #sat = hsv_img[:,:,1]
    #val = hsv_img[:,:,2]
    #
    #ret_sat,thresh_sat = cv2.threshold( sat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU ) 
    #ret_hue,thresh_hue = cv2.threshold( hue,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU ) 
    #mask = cv2.bitwise_and( thresh_hue,thresh_sat,mask=None )
    #
    #
    ## only keep the largest connected component
    #closed_mask = closing( mask,square(3) )
    #labeled_img = label(closed_mask, connectivity=2)

    ## leaf area should be the second largest number, with background being the most common
    #mode_label,count = ss.mode( labeled_img,axis=None )
    ## remove the most common label here
    #labeled_img_filtered = np.where( labeled_img==mode_label,np.nan,labeled_img )
    #mode_label,count = ss.mode( labeled_img_filtered,axis=None,nan_policy='omit' )
    #leaf_label = mode_label

    #leaf_mask = np.where( labeled_img==leaf_label,True,False )
    #print( 'leaf_mask.shape :',leaf_mask.shape )
    #low_s_bin = leaf_mask
    #img = cv2.cvtColor( img,cv2.COLOR_BGR2RGB )

    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################

    # --OSTU threshold
    # OSTU is a method to separate foreground from background
    low_s_thre = filters.threshold_otsu(np.array(low_s))
    #print( 'low_s_thre shape :',low_s_thre.shape )
    low_s_bin = low_s > low_s_thre  # row is y and col is x
    #print( 'low_s_bin shape :',low_s_bin.shape )

    # only keep the largest connected component
    low_s_bin = closing(low_s_bin, square(3))
    #print( 'low_s_bin shape :',low_s_bin.shape )
    labeled_img = label(low_s_bin, connectivity=2)
    #print( 'labeled_img shape :',labeled_img.shape )

    props = regionprops(labeled_img)
    area_list = np.zeros(len(props))
    #print( 'area_list shape :',area_list.shape )
    for i, reg in enumerate(props):  # i+1 is the label
        area_list[i] = reg.area

    # sort
    area_list = area_list.argsort()[::-1]
    #print( 'area_list shape :',area_list.shape )
    label_id = -1
    label_area = 0
    extent = 0
    for i in area_list:
        extent = props[i].extent
        if extent > ref_extent:
            label_id = i + 1
            label_area = props[i].area
            break
    if label_id == -1 or label_area < ref_area:
        print("extent:", extent)
        print("area", ref_area)

    assert label_id != -1, "(2) failed to find the leaf region in pre-processing!" \
                           "try to REDUCE 'ref_extent' a bit"

    # MP changed from assert label_area > ref_area to just print an error instead
    ####################################################################################################
    if label_area > ref_area:
        print('(1)WARNING')
        print("(1) failed to find the leaf region in pre-processing!\n Try to REDUCE 'ref_extent' a bit")
        print('(1) label_area : {}\tref_area : {}'.format(label_area,ref_area) )
    ####################################################################################################

    #assert label_area > ref_area, "fail to find the leaf region in pre-processing!" \
    #                              "try to REDUCE 'ref_extent' a bit"

    # TODO Why is this different?
    low_s_bin = labeled_img != label_id

    SlideName = slide.split('/')[-1]
    print( 'SlideName :',SlideName )
    slideNameMap_npz = SlideName.replace(".png", "_Map.npz")
    print( 'slideNameMap_npz :',slideNameMap_npz )
    slideNameMap_npz_pth = os.path.join(npz_dir_whole, slideNameMap_npz)
    print( 'slideNameMap_npz_pth :',slideNameMap_npz_pth )
    SegRes = load_npz(slideNameMap_npz_pth)
    SegRes = SegRes.todense()
    SegRes[low_s_bin] = 0
    save_npz(os.path.join(npz_dir_whole_post, slideNameMap_npz), csr_matrix(SegRes))

def rmbackground2( slide,leaf_mask_dir,npz_dir_whole,npz_dir_whole_post ):

    SlideName = slide.split('/')[-1]
    #print( 'SlideName :',SlideName )

    mskpth = os.path.join( leaf_mask_dir,'leaf_mask_'+SlideName )
    mskpth_p = mskpth.replace( '.png','.p' )
    leaf_mask = p.load( open(mskpth_p,'rb') )

    slideNameMap_npz = SlideName.replace(".png", "_Map.npz")
    #print( 'slideNameMap_npz :',slideNameMap_npz )
    slideNameMap_npz_pth = os.path.join(npz_dir_whole, slideNameMap_npz)
    #print( 'slideNameMap_npz_pth :',slideNameMap_npz_pth )
    SegRes = load_npz(slideNameMap_npz_pth)
    SegRes = SegRes.todense()

    #SegRes[low_s_bin] = 0
    #print( 'SegRes.shape :',SegRes.shape )
    #print( 'leaf_mask.shape ;',leaf_mask.shape )
    SegRes = np.where( leaf_mask==False,0,SegRes )

    #low_s_bin = leaf_mask

    save_npz(os.path.join(npz_dir_whole_post, slideNameMap_npz), csr_matrix(SegRes))

#TODO figure out what/where this is being saved

#SavePatchMap(npz_dir_whole_post, FigPath, slidename + "_Map.npz")
def SavePatchMap(npz_dir_whole_post, FigPath, MatName):
    FigName = MatName.replace('.npz', '.png')
    FigFile = os.path.join(FigPath, FigName)
    MatFile = os.path.join(npz_dir_whole_post, MatName)
    SegRes = load_npz(MatFile)
    SegRes = SegRes.todense()
    SegRes = vl2im(SegRes)
    Fig = Image.fromarray(SegRes.astype(dtype=np.uint8))
    del SegRes
    Fig.convert('RGB')
    Fig.save(FigFile, 'PNG')

#TODO what is this for?
def findN(cir_list):
    N = 0
    for i in cir_list:
        if i[0] > 20:
            N += 1

    return N




#TODO look at this?
#TODO postprocess to only record the largest (n_lesions) areas?
#def postprocess(cir_list, rect_list, N, imag, imagelst):
# N is number of circles?
def postprocess(cir_list, rect_list, N, imag, imagelst, root_dir,):
    print( '\nPostProcessing now!' )
    print( 'root_dir :',root_dir )

    # variables
    global count
    sought = [0, 0, 255]
    count += 1
    lesion = []
    area = []
    max_cir = []
#    root_dir = './result/'
    CoordPath = root_dir + "/PatchNet/npz/Coord"
    LesionArea = root_dir + "/PatchNet/npz/Area"

    result_dir = os.path.join( root_dir,'resultFiles' )
    print( 'result_dir :',result_dir )
    if not os.path.exists( result_dir ):
        os.makedirs( result_dir )

    result_file = os.path.join( result_dir,root_dir+'.xlsx' )
    print( '\tresult_file located :',result_file )

            #rect_list.append([w * h, (x, y), (x + w, y + h)])

    print( 'rect_list :',rect_list )

    max_rect = [p[1] for p in sorted(sorted(enumerate(rect_list), key=lambda x: x[1])[-N:], key=itemgetter(0))]
    lesion_id = [p[0]+1 for p in enumerate(max_rect)]
    print( 'max_rect :',max_rect )
    print( 'lesion_id :',lesion_id )
    del lesion_id[-1]

    # rects
    del max_rect[0]
    for x in max_rect:
        del x[0]
    for k in range(0, len(max_rect)):
        lesion.append(imag[max_rect[k][0][1]:max_rect[k][1][1], max_rect[k][0][0]:max_rect[k][1][0]])
    for d in range(0, len(lesion)):
        result = np.count_nonzero(np.all(lesion[d] == sought, axis=2))
        area.append(result)
    for j in range(0, len(max_rect)):
        cv2.putText(imag, str(j+1), (int(max_rect[j][0][0]), int(max_rect[j][0][1])), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

    if not os.path.exists(CoordPath):
        os.makedirs(CoordPath)
    filename = 'rects.txt'
    with open(os.path.join(CoordPath, filename), 'a') as file:
        file.write('image' + str(count) + ':' + str(max_rect) + '\n')
    print('\tbounding box coordinates saved')
    if not os.path.exists(LesionArea):
        os.makedirs(LesionArea)
    filename = 'lesion_area.txt'
    with open(os.path.join(LesionArea, filename), 'a') as file:
        file.write('image' + str(count) + ':' + str(area).strip('[]') + '\n')
    print('\tlesion area saved')

    # circles
    for i in range(0, N):
        max2 = max(cir_list, key=lambda x: x[0])
        cir_list.remove(max2)
        max_cir.append(max2)
    del max_cir[0]

    if not os.path.exists(CoordPath):
        os.makedirs(CoordPath)
    filename = 'circles.txt'
    with open(os.path.join(CoordPath, filename), 'a') as file:
        file.write('image' + str(count) + ': ' + str(max_cir) + '\n')
    print('\tcircle center coordinates and radius saved')

    # spreadsheet
    imname = imagelst[0].split('/')[-1]
    cameraID = imname[9:11]
    year = imname[12:16]
    month = imname[17:19]
    day = imname[20:22]
    hour = imname[23:27]

    l1 = cameraID
    l2 = year
    l3 = month
    l4 = day
    l5 = hour
    l6 = lesion_id
    l7 = area
    l8 = max_rect
    l9 = max_cir

    s1 = pd.Series(l1, name='Camera ID')
    s2 = pd.Series(l2, name='Year')
    s3 = pd.Series(l3, name='Month')
    s4 = pd.Series(l4, name='Day')
    s5 = pd.Series(l5, name='Time')
    s6 = pd.Series(l6, name='Lesion ID')
    s7 = pd.Series(l7, name='Area')
    s8 = pd.Series(l8, name='rect')
    s9 = pd.Series(l9, name='circle')
    # Create a Pandas dataframe from the data.
    df = pd.concat([s1, s2, s3, s4, s5, s6, s7, s8, s9], axis=1)

    spreadsheet = imagelst[0].split('/')[-2]
    print( '\tspreadsheet :',spreadsheet )
    #filename = spreadsheet + '.xlsx'
    #print( 'filename :',filename )
    #TODO

    if not os.path.exists( result_file ):
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(result_file, engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
    else:
        writer = pd.ExcelWriter(result_file, engine='openpyxl')
        # try to open an existing workbook
        writer.book = load_workbook(os.path.join(result_dir, filename))
        # copy existing sheets
        writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        # read existing file

        #TODO MP don't have excel on this computer
        reader = pd.read_excel(result_file)
        # write out the new sheet
        df.to_excel(writer, index=False, header=False, startrow=len(reader) + 1)
        writer.close()

class GUI(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        parent.title('Leaf Image Processor')
        parent.minsize(640, 400)

        tk.Label( text="Description of this experiment and image series",anchor='center' ).grid( column=5,row=0,sticky='NSEW' )

        # MP adding more labels for entry boxes
        ###################################################################################################################
        tk.Label( text='Save Name',anchor='center' ).grid( column=3,row=0,sticky=''  )
        tk.Label( text='Number of Lessions',anchor='center' ).grid( column=3,row=3,sticky=''  )

        self.parent.columnconfigure(0, weight=1)
        self.parent.columnconfigure(1, weight=1)
        self.parent.columnconfigure(2, weight=1)
        self.parent.columnconfigure(3, weight=1)
        self.parent.columnconfigure(4, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.rowconfigure(1, weight=1)
        self.parent.rowconfigure(2, weight=1)
        self.parent.rowconfigure(3, weight=1)
        ###################################################################################################################

        self.entryboxes()
        self.savebutton()
        self.selectbutton()
        self.RunwithCPUbutton()
        self.RunwithGPUbutton()
        self.nextbutton()
        self.previousbutton()


        self.num_lesions = StringVar()

    def entryboxes(self):
        self.e1 = tk.Text(root, height=4, width=40)
        self.e2 = tk.Entry(root)
        self.e1.grid( column=5,row=1,sticky='NSEW' )
        #TODO what is this 'e2' for? it is supposed to be input for GPU processing?
        #self.e2.grid( column=1,row=2,sticky='' )

        # MP adding more entry boxes
        ###################################################################################################################
        self.saveNameEntry = tk.Entry( root )
        self.saveNameEntry.insert( END,'default' )
        self.saveNameEntry.grid( column=3,row=1,sticky='' )
        self.numberOfLessionsEntry = tk.Entry( root )
        self.numberOfLessionsEntry.insert( END,'4' )
        self.numberOfLessionsEntry.grid( column=3,row=4,sticky='' )

        #tk.Label( text='Batch Start Time',anchor='center' ).grid( column=4,row=2,sticky=''  )
        #self.batchTimeVar = tk.StringVar( root )
        #self.batchTimeVar.set( 'Earliest in Batch' )
        #self.batchTimeMenu = tk.OptionMenu( root,self.batchTimeVar,'Earliest in Batch','0','1','2','3',
        #                '4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
        #                '21','22','23' )
        #self.batchTimeMenu.grid( column=4,row=3,sticky='' )

        tk.Label( text='Innoculation start Time',anchor='center' ).grid( column=4,row=0,sticky=''  )
        self.innocYearTimeVar = tk.Entry( root )
        self.innocYearTimeVar.insert( END,'year - 2020' )
        self.innocYearTimeVar.grid( column=4,row=1,sticky='' )

        self.innocMonthTimeVar = tk.Entry( root )
        self.innocMonthTimeVar.insert( END,'month - 1' )
        self.innocMonthTimeVar.grid( column=4,row=2,sticky='' )

        self.innocDayTimeVar = tk.Entry( root )
        self.innocDayTimeVar.insert( END,'day - 25' )
        self.innocDayTimeVar.grid( column=4,row=3,sticky='' )

        self.innocHourTimeVar = tk.Entry( root )
        self.innocHourTimeVar.insert( END,'hour - 1' )
        self.innocHourTimeVar.grid( column=4,row=4,sticky='' )

        self.selectsavedir()

        models_available = glob.glob( 'pytorch_models/*' )
        models_available_list = []
        for m in models_available:
            models_available_list.append( m.split('/')[-1] )
        tk.Label( text='Model Type',anchor='center' ).grid( column=1,row=3,sticky=''  )
        self.modelTypeVar = tk.StringVar( root )
        self.modelTypeVar.set( 'green.pth' )
        #self.modelTypeMenu = tk.OptionMenu( root,self.modelTypeVar,models_available )
        self.modelTypeMenu = tk.OptionMenu( root,self.modelTypeVar,*models_available_list )
        self.modelTypeMenu.grid( column=1,row=4,sticky='' )

        self.grid_columnconfigure(0,weight=1)
        self.grid_columnconfigure(1,weight=1)
        self.grid_columnconfigure(2,weight=1)
        self.grid_columnconfigure(3,weight=1)
        self.grid_columnconfigure(4,weight=1)
        self.grid_columnconfigure(5,weight=1)
        self.grid_rowconfigure(0,weight=1)
        self.grid_rowconfigure(1,weight=1) 
        self.grid_rowconfigure(2,weight=1) 
        self.grid_rowconfigure(3,weight=1) 
        ###################################################################################################################
        #TODO add a border around GUI

    def setImageFrame( self, ):
        self.imageFrame = tk.Frame()
        self.selectbutton()

    def selectbutton(self):
        self.selectbutton = tk.Button( master=self.imageFrame,text='Select Images',command=self.openimage )
        self.selectbutton.grid( column=1,row=0 )



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
        print('description saved')

    def savebutton(self):
        self.savebutton = tk.Button(text='save', command=self.save,anchor='center' )
        self.savebutton.grid(column=5, row=2)

    def selectbutton(self):
        self.selectbutton = tk.Button(text='Select Images', command=self.openimage)
        self.selectbutton.grid( column=1,row=0)

    def selectsavedir( self ):
        self.saveDir = 'None'
        self.selectsavedir = tk.Button( text='Set Results Directory',command=self.setSaveDir )
        self.selectsavedir.grid( column=3,row=2)

    #def RunwithCPUbutton(self):
    #    self.RunwithCPUbutton = tk.Button(text='Run with CPU', command=lambda: self.process(""))
    #    self.RunwithCPUbutton.grid( column=1,row=3 )

    def RunwithCPUbutton(self):
        self.RunwithCPUbutton = tk.Button(text='Run with CPU',command=self.process)
        self.RunwithCPUbutton.grid( column=5,row=3 )

    #def RunwithGPUbutton(self):
    #        #self.RunwithGPUbutton = tk.Button(text='Run with GPU', command=lambda: self.process(self.e2.get()), state=tk.DISABLED)
    #    self.RunwithGPUbutton = tk.Button(text='Run with GPU', command=lambda: self.process( self.e2.get()) ))
    #    self.RunwithGPUbutton.grid( column=1,row=4 )

    def RunwithGPUbutton(self):
            #self.RunwithGPUbutton = tk.Button(text='Run with GPU', command=lambda: self.process(self.e2.get()), state=tk.DISABLED)
        self.RunwithGPUbutton = tk.Button(text='Run with GPU', command=self.process )
        self.RunwithGPUbutton.grid( column=5,row=4 )

    def move(self, delta):
        global current
        if not (0 <= current + delta < len(imagelst)):
            return
        else:
            current += delta
            pic = Image.open(imagelst[current])
            pic = pic.resize((230, 230), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(pic)
            self.label = Label(image=img)
            self.label.image = img
            self.label.grid(column=1, row=1, pady=10)

    def nextbutton(self):
        self.nextbutton = tk.Button( root,text='Next',command=lambda: self.move(+1) )
        self.nextbutton.grid(column=2, row=2)

    def previousbutton(self):
        self.previousbutton = tk.Button(text='Previous', command=lambda: self.move(-1))
        self.previousbutton.grid(column=0, row=2)

    def sortimages( self,filenames ):
        imagemat = []
        for i in filenames:
            imname = i.split('/')[-1]
            cameraID = int( imname[9:11] )
            year = int( imname[12:16] )
            month = int( imname[17:19] )
            day = int( imname[20:22] )
            hour = int( int( imname[23:27] ) / 100 )
            imagemat.append( [imname,i,cameraID,year,month,day,hour] )

        self.imageDF = pd.DataFrame( imagemat,columns=['name','filename','cameraID','year','month','day','hour',] )
        self.imageDF.sort_values( by=['cameraID','year','month','day','hour'] )
        imagelst = self.imageDF['filename']
        return imagelst

    def openimage(self):
        global imagelst
        global imglst
        self.filenames = filedialog.askopenfilenames(filetypes=( ('all files','*.*'),('png files','*.png'),('jpeg files','*.jpeg') ), initialdir='/', title='Select Image')

        imagelst = self.sortimages( self.filenames )

        #for i in self.filenames:
        #    imagelst.append(i)

        pic = Image.open(imagelst[0])
        pic = pic.resize((230, 230), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(pic)
        self.label = Label(image=img)
        self.label.image = img
        self.label.grid(column=1, row=1)

    def setSaveDir(self):
        self.saveDir = filedialog.askdirectory( initialdir='/',title='Select Save Directory' )

    def numLessionsFilter( self,new_labeled_img ):
        label_props_area_list = np.zeros( shape=[len(new_labeled_img),2] )
        label_props = regionprops( new_labeled_img )  
        for l,prop in enumerate(label_props):
            label_props_area_list[ l,0 ] = prop.label
            label_props_area_list[ l,1 ] = prop.area
        sorted_area_list = label_props_area_list[ np.argsort( label_props_area_list[:,1] ) ]
        areas_to_remove = sorted_area_list[ :(-1*self.num_lesions),: ]
    
        for area in range( len( areas_to_remove ) ):
            new_labeled_img = np.where( new_labeled_img==areas_to_remove[area,0],0,new_labeled_img )

        return new_labeled_img

    #TODO making n='' for now
    def process(self,n=''):

        os.environ["CUDA_VISIBLE_DEVICES"] = n

        num_lesions = self.numberOfLessionsEntry.get()
        if num_lesions == '4':
            print( '\nUsing 4 lesions as default value!')
        self.num_lesions = int( num_lesions )


        root_dir = self.saveNameEntry.get()
        if root_dir == 'default':
            print( '\nEnter a save file name first!\nusing "default" for now.')

        # save description text
        self.save()

        saveDir = self.saveDir
        print( 'saveDir :',saveDir )
        if saveDir != 'None':
            root_dir = os.path.join( saveDir,self.saveNameEntry.get() )
        print( 'root_dir :',root_dir )

        self.model_name = self.modelTypeVar.get()

        mask_dir = os.path.join( root_dir,'masks' )
        if not os.path.exists( mask_dir ):
            os.makedirs( mask_dir )

        leaf_mask_dir = os.path.join( mask_dir,'leaf_masks' )
        if not os.path.exists( leaf_mask_dir ):
            os.makedirs( leaf_mask_dir )

        lesion_mask_dir = os.path.join( mask_dir,'lesion_masks' )
        if not os.path.exists( lesion_mask_dir ):
            os.makedirs( lesion_mask_dir )

        result_dir = os.path.join( root_dir,'resultFiles' )
        print( 'result_dir :',result_dir )
        if not os.path.exists( result_dir ):
            os.makedirs( result_dir )

        patch_dir = os.path.join( root_dir,'patches' )
        if not os.path.exists( patch_dir ):
            os.makedirs( patch_dir )

        patch_image_dir = os.path.join( patch_dir,'images' )
        if not os.path.exists( patch_image_dir ):
            os.makedirs( patch_image_dir )

        log_dir = os.path.join( root_dir,'log' )
        if not os.path.exists( log_dir ):
            os.makedirs( log_dir )

        log_pth = os.path.join( log_dir,'log.log' )

        txt_dir = os.path.join( root_dir,'txt' )
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        # I'm not exactly sure what is being saved in these patchnet directories
        ###################################################################
        patchnet_dir = os.path.join( root_dir,'PatchNet' )

        npz_dir = os.path.join( patchnet_dir,'npz' )
        if not os.path.exists(npz_dir):
            os.makedirs(npz_dir)

        map_dir = os.path.join( patchnet_dir,'map' )
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)

        log_dir = os.path.join( patchnet_dir,'logfiles' )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        npz_dir_whole = os.path.join( npz_dir,'whole_npz' )
        npz_dir_whole_post = os.path.join( npz_dir,'whole_npz_post' )
        FigPath = os.path.join( result_dir,'whole_fig_post' )
        PostPath = os.path.join( result_dir,'whole_fig_post_rm' )
        imgsWLessions_dir = os.path.join( result_dir,'imgsWLessions' )
        if not os.path.exists(imgsWLessions_dir):
            os.makedirs(imgsWLessions_dir)
        if not os.path.exists(npz_dir_whole_post):
            os.makedirs(npz_dir_whole_post)
        if not os.path.exists(FigPath):
            os.makedirs(FigPath)
        if not os.path.exists(PostPath):
            os.makedirs(PostPath)
        ###################################################################

        # More path stuff
        ###############################################################################################
        # merge npz
        # maybe change this part?
        whole_npz_dir = os.path.join( npz_dir,'whole_npz' )
        if not os.path.exists( whole_npz_dir ):
            os.makedirs( whole_npz_dir )

        ###############################################################################################

        ###################################################################
        # make directory to hold pytorch models
        model_dir = os.path.join( 'pytorch_models' )
        if not os.path.exists( model_dir ):
            os.makedirs( model_dir )
        # choose pytorch model based off of model type
        model_id = self.model_name
        model_pth = os.path.join( model_dir,( model_id ) )
        RESTORE_FROM = model_pth
        ###################################################################
        
        #self.batchTimeVar.set( 'Earliest in Batch' )
        #self.batchStartTime = self.batchTimeVar.get()
        #if self.batchStartTime == 'Earliest in Batch':
        #    self.startTime = self.imageDF.at[0,'hour']
        #else:
        #    self.startTime = self.batchStartTime
        #self.imageDF[ 'startTime' ] = self.startTime

        for i in range( self.num_lesions ):
            area_str = 'l'+str(i+1)+'_area'
#            radius_str = 'l'+str(i)+'_radius'
            cenX_str = 'l'+str(i+1)+'_centerX'
            cenY_str = 'l'+str(i+1)+'_centerY'
            startX_str = 'l'+str(i+1)+'_xstart'
            startY_str = 'l'+str(i+1)+'_xend'
            endX_str = 'l'+str(i+1)+'_ystart'
            endY_str = 'l'+str(i+1)+'_yend'
            self.imageDF[ area_str ] = ''
#            self.imageDF[ radius_str ] = ''
            self.imageDF[ cenX_str ] = ''
            self.imageDF[ cenY_str ] = ''
            self.imageDF[ startX_str ] = ''
            self.imageDF[ startY_str] = ''
            self.imageDF[ endX_str ] = ''
            self.imageDF[ endY_str ] = ''

        self.imageDF[ 'InnoculationYear' ] = int( self.innocYearTimeVar.get() )
        self.imageDF[ 'InnoculationMonth' ] = int( self.innocMonthTimeVar.get() )
        self.imageDF[ 'InnoculationDay' ] = int( self.innocDayTimeVar.get() )
        self.imageDF[ 'InnoculationHour' ] = int( self.innocHourTimeVar.get() )

        hours_elapsed = []
        end_datetime = dt.datetime( int(self.innocYearTimeVar.get()),
                        int(self.innocMonthTimeVar.get()),
                        int(self.innocDayTimeVar.get()),
                        int(self.innocHourTimeVar.get()) 
                        ) 

        for df_index,df_row in self.imageDF.iterrows():
            start_datetime = dt.datetime( df_row['year'],df_row['month'],df_row['day'],df_row['hour'] ) 
            time_diff = end_datetime - start_datetime
            secs_diff = time_diff.total_seconds()
            hours_diff = np.divide( secs_diff,3600 )
            hours_elapsed.append( int( hours_diff ) )

        self.imageDF[ 'HoursElapsed' ] = hours_elapsed

        #while imagelst:
        #for tip in self.imageDF['filename']: 
        for df_index,df_row in self.imageDF.iterrows(): 
            # input
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print( '\tUsing device: ',device )
            if device.type == 'cuda':
                print( '\t',torch.cuda.get_device_name(0) )
            modelname = self.modelTypeVar.get()
            # setting
            patch_size = 512
            #TODO
            #patch_size = 256 
            INPUT_SIZE = (patch_size, patch_size)
            overlap_size = 64
            #TODO look into this
            ref_area = 10000  # pre-processing
            ref_extent = 0.6  # pre-processing
            rm_rf_area = 5000  # post-processing
            ref_ecc = 0.92  # post-processing
            BATCH_SIZE = 32

            #test_img_pth = imagelst[0] # name of the image, but starts with a '/'
            test_img_pth = df_row[ 'filename' ]
            #print( 'test_img_pth :',test_img_pth )
            filename = test_img_pth.split('/')[-1] # name of the image
            #print( 'filename :',filename )
            slidename = filename[:-4] # slidename is just the name of the image w/o extension

            patch_mask_slide_dir = os.path.join( lesion_mask_dir,slidename )
            if not os.path.exists( patch_mask_slide_dir ):
                os.makedirs( patch_mask_slide_dir )

            log = open(log_pth, 'w')
            log.write(test_img_pth + '\n')

            # divide testing image into patches
            process_tif(test_img_pth, filename, patch_image_dir, log, patch_size,
                        overlap_size, ref_area=ref_area, ref_extent=ref_extent, leaf_mask_dir=leaf_mask_dir ) #TODO error here with finding ref_extent

            txtname = slidename + ".txt"
            txtfile = os.path.join(txt_dir, txtname)
            txt = open(txtfile, 'w')
            SlideDir = os.path.join(patch_image_dir, slidename)
            PatchList = os.listdir(SlideDir)
            for PatchName in PatchList:
                PatchFile = os.path.join(SlideDir, PatchName)
                txt.write(PatchFile + '\n')
            txt.close()

            #################################################
            # testing
            IMG_MEAN = np.array((62.17962105572224, 100.62603236734867, 131.60830906033516), dtype=np.float32)
            #TODO do I need this? below()
            #DATA_DIRECTORY = patch_image_dir
            data_list_pth = txt_dir
            NUM_CLASSES = 2

#############################################################################################################################
#############################################################################################################################
            preprocess_start_time = time.time()
            print("\nModel restored from:", RESTORE_FROM)
            # TODO What LogName to use?
            LogName = "Test_HeatMap_log.txt"
            LogFile = os.path.join(log_dir, LogName)
            print( 'LogFile :',LogFile )
            log = open(LogFile, 'w')
            log.writelines('batch size:' + str(BATCH_SIZE) + '\n')
            log.writelines(data_list_pth + '\n')
            log.writelines('restore from ' + RESTORE_FROM + '\n')
            model = UNet(NUM_CLASSES)
            model = nn.DataParallel(model)
            model.to(device)
            saved_state_dict = torch.load(RESTORE_FROM, map_location=lambda storage, loc: storage)
            num_examples = saved_state_dict['example']
            print("\tusing running mean and running var")
            log.writelines("using running mean and running var\n")
            model.load_state_dict(saved_state_dict['state_dict'])
            model.eval()
            log.writelines('preprocessing time: ' + str(time.time() - preprocess_start_time) + '\n')
            print('\nProcessing ' + slidename)
            log.writelines('Processing ' + slidename + '\n')
            TestTxt = os.path.join(data_list_pth, slidename + '.txt')
            testloader = data.DataLoader(LEAFTest(TestTxt, crop_size=INPUT_SIZE, mean=IMG_MEAN),
                                         batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
            #TODO maybe change these names?
            TestNpzPath = os.path.join(npz_dir, slidename)
            TestMapPath = os.path.join(map_dir, slidename)
            if not os.path.exists(TestNpzPath):
                os.mkdir(TestNpzPath)
            if not os.path.exists(TestMapPath):
                os.mkdir(TestMapPath)
            batch_time = AverageMeter()
            with torch.no_grad():
                end = time.time()
                for index, (image, name) in enumerate(testloader):
                    output = model(image).to(device)
                    del image
                    Softmax = torch.nn.Softmax2d()
                    pred = torch.max(Softmax(output), dim=1, keepdim=True)
                    del output
                    #p.dump( pred[0],open('pred0.p','wb') )
                    #p.dump( pred[1],open('pred1.p','wb') )

                    for ind in range(0, pred[0].size(0)):
                        prob = torch.squeeze(pred[0][ind]).data.cpu().numpy()
                        prob = coo_matrix(prob)
                        if len(prob.data) == 0:
                            continue
                        mapname = name[ind].replace('.jpg', '_N' + str(num_examples) + '_MAP.npz')
                        mapfile = os.path.join(TestMapPath, mapname)
                        save_npz(mapfile, prob.tocsr())

                        msk = torch.squeeze(pred[1][ind]).data.cpu().numpy()
                        # MP save each individual patch map
                        ##################################################################
                        msk_copy = msk.copy()
                        msk = coo_matrix(msk)

                        if len(msk.data) == 0:
                            continue

                        patch_msk_name = name[ind].replace( '.jpg','_mask.jpg' )
                        patch_msk_pth = os.path.join( patch_mask_slide_dir,patch_msk_name )
                        plt.imsave( patch_msk_pth,msk_copy )
                        ##################################################################
                        npzname = name[ind].replace('.jpg', '_N' + str(num_examples) + '_MSK.npz')
                        npzfile = os.path.join(TestNpzPath, npzname)
                        save_npz(npzfile, msk.tocsr())

                    batch_time.update(time.time() - end)
                    end = time.time()

                    if index % 10 == 0:
                        print('Test:[{0}/{1}]\t'
                              'Time {batch_time.val:.3f}({batch_time.avg:.3f})'
                              .format(index, len(testloader), batch_time=batch_time))

#############################################################################################################################
#############################################################################################################################

            print('\tThe total test time for ' + slidename + ' is ' + str(batch_time.sum))
            log.writelines('batch num:' + str(len(testloader)) + '\n')
            log.writelines('The total test time for ' + slidename + ' is ' + str(batch_time.sum) + '\n')
            
            # MP don't know if this is used at all?
            #logfile = open(os.path.join(logdir, 'merge_npz_' + model_id + '.log'), 'w')
            slide_map_name_npz = slidename + '_Map.npz'
            slide_map_name_png = slidename + '_Map.png'

            img = spm.imread(test_img_pth)
            width, height = img.shape[1], img.shape[0]
            final = merge_npz(os.path.join(npz_dir), slidename,
                              os.path.join( whole_npz_dir,slide_map_name_npz ),
                              int(width), int(height))
            # if not os.path.exists(CRFFigPath):
            #     os.makedirs(CRFFigPath)
            print("\nremove background")
            # TODO replace background removal. Should just use the mask we already saved
            #rmbackground(test_img_pth, npz_dir_whole, npz_dir_whole_post, ref_extent=ref_extent, ref_area=ref_area)
            rmbackground2( test_img_pth,leaf_mask_dir,npz_dir_whole,npz_dir_whole_post ) 
            SavePatchMap(npz_dir_whole_post, FigPath, slidename + "_Map.npz")

            pred_img_pth = os.path.join( FigPath,slide_map_name_png )
            # crf_img_name = slidename+"_Map.png"
            # crf_img_pth = os.path.join(CRFFigPath, crf_img_name)
            # CRFs(test_img_pth, pred_img_pth, crf_img_pth)
            post_img_pth = os.path.join( PostPath,slidename + "_Map.png" )
            #print( 'post_img_pth :',post_img_pth )
            #TODO what is pred_img_pth vs test_img_pth
            print( 'pred_img_pth :',pred_img_pth )
            img = spm.imread(pred_img_pth)
            img = im2vl(img) # This returns (512x512) with positive areas = 1 and background = 0
            print( 'img :',img )
            print( 'max(img) :',np.max(img) )

            #####################################################################################################################
            # floolfill stuff
            #####################################################################################################################
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
            cv2.imwrite( 'floodfillTest.png',filled_holes )
            #####################################################################################################################
            filled_holes[ filled_holes!=0 ] = 1
            img = filled_holes
            #####################################################################################################################

            img_close = closing(img, square(3))
            labeled_img = label(img_close, connectivity=2)
            new_labeled_img = remove_small_objects(labeled_img, min_size=rm_rf_area, connectivity=2)
            # remove non-circle region
            props = regionprops(new_labeled_img)
            for i, reg in enumerate(props):
                if reg.eccentricity > ref_ecc:
                    new_labeled_img[new_labeled_img == reg.label] = 0
            #new_img = np.asarray(new_labeled_img != 0, dtype=np.uint8)
            
            # Filter to num_lesions here
            new_labeled_img = self.numLessionsFilter( new_labeled_img )

            new_img = np.asarray(new_labeled_img != 0, dtype=np.uint8)

            new_img = vl2im(new_img)
            imsave(post_img_pth, new_img)

            # Draw bounding box
            cv2.namedWindow(test_img_pth, cv2.WINDOW_NORMAL)
            #imag = cv2.pyrDown(cv2.imread(post_img_pth, cv2.IMREAD_UNCHANGED))
            leaf_img = cv2.imread( test_img_pth )
            imag = cv2.imread(post_img_pth, cv2.IMREAD_UNCHANGED)
            gray = cv2.cvtColor( imag,cv2.COLOR_BGR2GRAY )

            # Overlapping over original leaf image here
            ret,gray_mask = cv2.threshold( gray,200,255,cv2.THRESH_BINARY_INV )
            gray_mask_inv = cv2.bitwise_not( gray_mask )
            leaf_bgr = cv2.bitwise_and( leaf_img,leaf_img,mask=gray_mask_inv )
            lesion_fg = cv2.bitwise_and( imag,imag,mask=gray_mask )
            im_to_write = cv2.add( leaf_bgr,lesion_fg )

            blurred = cv2.GaussianBlur( gray,(5,5),0 )
            #TODO fix this so that it isn't a loop
            r,c = blurred.shape
            for row in range( r ):
                for col in range( c ):
                    if blurred[row,col] != 255 and blurred[row,col] != 76:
                        blurred[row,col] = 76
            blurred_bit = cv2.bitwise_not( blurred )

            _,labels,stats,centroid = cv2.connectedComponentsWithStats( blurred_bit )
            #print( 'stats :',stats )
            #print( 'centroid :',centroid )
            #p.dump( labels,open('labels.p','wb') )


            ## threshold image
            #ret, threshed_img = cv2.threshold(cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY),
            #                                  127, 255, cv2.THRESH_BINARY)
            ##TODO What are these contours?
            ## find contours and get the external one
            #contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # with each contour, draw boundingRect in green
            # a minEnclosingCircle in blue
            rect_list = []
            cir_list = []

            print( '\nPostProcessing now!' )


    ########################################################################################################
    ########################################################################################################
        #def lesionOrderCheck( self,df,contours,row ):
            contour_arr = np.zeros( shape=(len(stats),8) )
            #for i,c in enumerate(contours):
            for i in range( len(stats) ):
                # ignore the first entry because it is the background
                if i > 0:
                    # get the bounding rect
                    x = stats[i,0]
                    y = stats[i,1]
                    w = stats[i,2]
                    h = stats[i,3]
    
                    contour_arr[i,:5] = [ w*h,x,y,x+w,y+h ]
                    rect_list.append([w * h, (x, y), (x + w, y + h)])
    
    
                    #(x, y), radius = cv2.minEnclosingCircle(c)
                    # convert all values to int
                    #center = (int(x), int(y))
                    #radius = int(radius)
    
                    cx = centroid[i,0]
                    cy = centroid[i,1]
                    contour_arr[i,5:-1] = [ int(cx),int(cy) ]
                    #contour_arr[i,7] = np.count_nonzero( labels==i )
                    contour_arr[i,7] = stats[i,4]
                    #cir_list.append([radius, center])
    
            sought = [0, 0, 255]
            lesion = []
            #area = []
            #for k in range(0, len(contour_arr)):
            #        #lesion.append( blurred[ int(contour_arr[k,1]):int(contour_arr[k,3]),int(contour_arr[k,2]):int(contour_arr[k,4]) ] )
            #    labeled_area = blurred[ int(contour_arr[k,1]):int(contour_arr[k,3]),int(contour_arr[k,2]):int(contour_arr[k,4]) ]
            #    labeled_area = np.where( labeled_area==76,1,0 )
            ##for d in range(0, len(contour_arr)):
            #        #result = np.count_nonzero(np.all(lesion[d] == sought, axis=2))
            #    result = np.sum( labeled_area )
            #    area.append(result)
            #    contour_arr[k,7] = result

            # sort the contours by radius to choose the num_lesions largest contours

            # Take the n_lesion largest areas, then sort by x+y locations
            #print( 'contour_arr :',contour_arr )
            contour_arr = contour_arr[ contour_arr[:,7].argsort() ][-self.num_lesions:,:]
            contour_arr = contour_arr[ contour_arr[:,5].argsort() ]
            contour_arr = contour_arr[ contour_arr[:,6].argsort(kind='mergesort') ]
            #print( 'contour_arr :',contour_arr )
    
            # check if lesions are in the same order
            lesion_bounds = []
            if df_index > 0:
                prev_cameraID = self.imageDF.loc[ df_index-1,'cameraID' ]
                prev_year = self.imageDF.loc[ df_index-1,'year' ]
                prev_month = self.imageDF.loc[ df_index-1,'month' ]
                prev_day = self.imageDF.loc[ df_index-1,'day' ]

                cameraID = self.imageDF.loc[ df_index,'cameraID' ]
                year = self.imageDF.loc[ df_index,'year' ]
                month = self.imageDF.loc[ df_index,'month' ]
                day = self.imageDF.loc[ df_index,'day' ]

                ##################################################################################################
                # TODO check here for lesion disapearing
                # contours_ordered is zero and filled with the lesion info if the lesion center falls within
                # the previous lesion bounds
                ##################################################################################################
                contours_ordered = np.zeros( contour_arr.shape )
                # TODO check if it is only previous hour?
                if prev_cameraID==cameraID and prev_year==year and prev_month==month and prev_day==day:
                    for l in range(self.num_lesions):
                        lesion_bounds.append( [ self.imageDF.loc[df_index-1,'l'+str(l+1)+'_xstart'],
                                self.imageDF.loc[df_index-1,'l'+str(l+1)+'_xend'],
                                self.imageDF.loc[df_index-1,'l'+str(l+1)+'_ystart'],
                                self.imageDF.loc[df_index-1,'l'+str(l+1)+'_yend']
                                ] )
                    # put the contours in the correct order so that they match the lesion numbers
                    lesion_bounds = np.array( lesion_bounds )
                    #print( 'lesion_bounds :',lesion_bounds )
                    for i in range( len(contour_arr) ):
                        for j in range( int(self.num_lesions) ):
                            if contour_arr[i,5] > lesion_bounds[j,0]\
                                            and contour_arr[i,5] < lesion_bounds[j,1]\
                                            and contour_arr[i,6] > lesion_bounds[j,2]\
                                            and contour_arr[i,6] < lesion_bounds[j,3]:
                                contours_ordered[i,:] = contour_arr[j,:]
                else:
                    contours_ordered = contour_arr
            else:
                contours_ordered = contour_arr
            #print( 'contour_arr[1] :',contour_arr )
            #print( 'contours_ordered :',contours_ordered )

            # write the lesions to the dataframe
            #print( 'df_index :',df_index )
            for l in range( self.num_lesions ):
                area_str = 'l'+str(l+1)+'_area'
                self.imageDF.at[ df_index,area_str ] = contours_ordered[l,7]
                #print( 'contours_ordered[l,8] :',contours_ordered[l,7] )
                self.imageDF.at[ df_index,'l'+str(l+1)+'_centerX' ] = contours_ordered[l,5]
                self.imageDF.at[ df_index,'l'+str(l+1)+'_centerY' ] = contours_ordered[l,6]
                self.imageDF.at[ df_index,'l'+str(l+1)+'_xstart' ] = contours_ordered[l,1]
                self.imageDF.at[ df_index,'l'+str(l+1)+'_xend' ] = contours_ordered[l,3]
                self.imageDF.at[ df_index,'l'+str(l+1)+'_ystart' ] = contours_ordered[l,2]
                self.imageDF.at[ df_index,'l'+str(l+1)+'_yend' ] = contours_ordered[l,4]

            for j in range(0, len(contours_ordered)):
                cv2.putText(im_to_write, str(j+1), (int(contours_ordered[j,5]), int(contours_ordered[j,6])), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 5)
                start = (int(contours_ordered[j,1]),int(contours_ordered[j,2]))
                end = (int(contours_ordered[j,3]),int(contours_ordered[j,4]))
                color = (0,0,0)
                thickness = 2
                #TODO check rectangles
                cv2.rectangle( im_to_write,start,end,color,thickness )
    

    
    
    ########################################################################################################
    ########################################################################################################

            #for c in contours:
            #    # get the bounding rect
            #    x, y, w, h = cv2.boundingRect(c)
            #    # draw a green rectangle to visualize the bounding rect
            #    cv2.rectangle(imag, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #    rect_list.append([w * h, (x, y), (x + w, y + h)])

            #    # finally, get the min enclosing circle
            #    (x, y), radius = cv2.minEnclosingCircle(c)
            #    # convert all values to int
            #    center = (int(x), int(y))
            #    radius = int(radius)
            #    # and draw the circle in blue
            #    imag = cv2.circle(imag, center, radius, (255, 0, 0), 2)
            #    cir_list.append([radius, center])


            #TODO just add to df here

            #print( 'cir_list :',cir_list )
            #print( 'rect_list :',rect_list )
            #print( 'findN(cir_list) :',findN(cir_list) )

            #postprocess( cir_list,rect_list,self.num_lesions,imag,imagelst,root_dir, )

            #cv2.drawContours(imag, contours, -1, (255, 255, 0), 1)

            #TODO is this where the images are shown?
#            cv2.imshow(test_img_pth, imag)
            #print( "self.imageDF[df_index,'name'] :",self.imageDF.loc[df_index,'name'] )
            img_sav_pth = os.path.join( imgsWLessions_dir,self.imageDF.loc[df_index,'name'] )
            #print( 'img_sav_pth :',img_sav_pth )
            cv2.imwrite( img_sav_pth,im_to_write )
            cv2.imshow( test_img_pth,im_to_write )


            #del imagelst[0]

        result_dir = os.path.join( root_dir,'resultFiles' )
        if not os.path.exists( result_dir ):
            os.makedirs( result_dir )


        result_file = os.path.join( result_dir,self.save_file_name+'.xlsx' )
        pickle_file = os.path.join( result_dir,self.save_file_name+'.p' )
        print( '\tresult_file located :',result_file )

        description = self.e1.get('1.0', END)
        self.imageDF['description'] = description

        p.dump( self.imageDF,open(pickle_file,'wb') )
        clean_df = self.imageDF.copy()

        for col in clean_df.columns:
            if 'center' in str(col) or 'start' in str(col) or 'end' in str(col):
                clean_df = clean_df.drop( columns=[col] )

        if not os.path.exists( result_file ):
            writer = pd.ExcelWriter(result_file, engine='xlsxwriter')
            clean_df.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.save()

        print( '\n****************************************************************************' )
        print( '***********************************DONE*************************************' )
        print( '****************************************************************************' )

if __name__ == "__main__":
    root = tk.Tk()
    GUI(root)
    root.mainloop()
