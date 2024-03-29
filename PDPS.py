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
import model.u_net2 as u_net

import torch.nn as nn
from PIL import Image
import datetime as dt
import glob
import warnings

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
        defaultFont.configure( size=30 )
        defaultTextFont.configure( size=20 )

        self.parent.grid_rowconfigure( 0,weight=3 )
        self.parent.grid_rowconfigure( (1,2),weight=1 )
        self.parent.grid_columnconfigure( (0,1,2,3),weight=1 )

        bg1 = '#B6CA53'
        fg1 = '#C4D473'
        bg2 = '#DDD755'
        fg2 = '#E7E288'
        bg3 = '#88B668'
        fg3 = '#CBDFBD'
        self.setImageFrame(bg=bg1,fg=fg1) 
        self.setModelFrame(bg=bg2,fg=fg2)
        self.setLesionFrame(bg=bg3,fg=fg3)
        self.setSaveFolderFrame(bg=bg3,fg=fg3)
        self.setInnocFrame(bg=bg1,fg=fg1)
        self.setDescFrame(bg=bg2,fg=fg2)
        self.setDirFrame(bg=bg2,fg=fg2)
        self.setSegmentationFrame(bg=bg1,fg=fg1)
        self.setRunFrame(bg=bg3,fg=fg3)

    #def setImageFrame( self, ):
    #    self.imageFrame = tk.Frame( root,bg='#65C1E8',borderwidth=15,relief='ridge' )
    #    self.nextbutton()
    #    self.previousbutton()

    #    self.imageFrame.grid_rowconfigure( (0,1,3),weight=1 )
    #    self.imageFrame.grid_rowconfigure( (2,),weight=3 )
    #    self.imageFrame.grid_columnconfigure( (0,1),weight=1 )
    #    self.imageFrame.grid( row=0,column=0,sticky='nsew' )

    #    self.selectbutton()

    def setImageFrame( self,bg,fg ):
        self.imageFrame = tk.Frame( root,bg=bg,borderwidth=10,relief='ridge' )

        self.imageFrame.grid_rowconfigure( (0),weight=5 )
        #self.imageFrame.grid_rowconfigure( (1),weight=1 )
        self.imageFrame.grid_columnconfigure( (0),weight=1 )
        self.imageFrame.grid( row=0,column=0,sticky='nsew' )

        self.selectbutton(fg)

    #def setModelFrame( self, ):
    #    self.modelFrame = tk.Frame( root,bg='#D85B63',borderwidth=15,relief='ridge' )

    #    self.modelTypeVar = tk.StringVar( self.modelFrame )
    #    self.modelTypeVar.set( 'LEAF_UNET_BROWN_SEP22.pth' )
    #    models_available_list = self.getModelsAvailable()
    #    #self.modelTypeMenu = tk.OptionMenu( self.modelFrame,self.modelFrame,self.modelTypeVar,*models_available_list )
    #    self.modelTypeMenu = tk.OptionMenu( self.modelFrame,self.modelTypeVar,*models_available_list )
    #    self.modelTypeMenu.grid( row=1,column=0 )

    #    self.modelFrame.grid_rowconfigure( (0,1),weight=1 )
    #    self.modelFrame.grid_columnconfigure( 0,weight=1 )
    #    self.modelFrame.grid( row=1,column=0,sticky='nsew' )

    def setModelFrame( self,bg,fg ):
        self.modelFrame = tk.Frame( root,bg=bg,borderwidth=10,relief='ridge' )

        self.modelTypeVar = tk.StringVar( self.modelFrame )
        self.modelTypeVar.set( 'Leaf DICE' )
        models_available_list = self.getModelsAvailable()
        #self.modelTypeMenu = tk.OptionMenu( self.modelFrame,self.modelFrame,self.modelTypeVar,*models_available_list )
        self.modelTypeMenu = tk.OptionMenu( self.modelFrame,self.modelTypeVar,*models_available_list )
        self.modelTypeMenu.config(bg=fg)
        self.modelTypeMenu.grid( row=0,column=0 )

        self.modelFrame.grid_rowconfigure( (0),weight=1 )
        self.modelFrame.grid_columnconfigure( 0,weight=1 )
        self.modelFrame.grid( row=1,column=0,sticky='nsew' )

    #def setDirFrame( self, ):
    #    self.dirFrame = tk.Frame( root,bg='#D680AD',borderwidth=15,relief='ridge' )
    #    folderLabel = tk.Label( self.dirFrame,text='Set Save Folder Name :',anchor='center' )
    #    folderLabel.grid( row=0,column=0,sticky=''  )
    #    self.saveNameEntry = tk.Entry( self.dirFrame )
    #    self.saveNameEntry.insert( END,'default' )
    #    self.saveNameEntry.grid( row=1,column=0,sticky='' )

    #    separator = Separator( self.dirFrame,orient='horizontal' )
    #    separator.grid( row=2,sticky='ew' )

    #    self.dirFrame.grid_rowconfigure( (0,1,2,3,4,5),weight=1 )
    #    self.dirFrame.grid_columnconfigure( (0),weight=1 )
    #    self.dirFrame.grid( row=0,column=1,sticky='nsew' )

    #    self.selectsavedir()

    def setSaveFolderFrame( self,bg,fg ):
        self.saveFolderFrame = tk.Frame( root,bg=bg,borderwidth=10,relief='ridge' )
        #folderLabel = tk.Label( self.saveFolderFrame,text='Set Save Folder:',anchor='center' )
        folderLabel = tk.Label( self.saveFolderFrame,text='Save Folder:',bg=bg )
        folderLabel.grid( row=0,column=0,sticky='s'  )
        self.saveNameEntry = tk.Entry( self.saveFolderFrame,width=15 )
        self.saveNameEntry.insert( END,'default' )
        self.saveNameEntry.grid( row=1,column=0,sticky='n' )
        self.saveFolderFrame.grid_rowconfigure( (0,1),weight=1 )
        self.saveFolderFrame.grid_columnconfigure( (0),weight=1 )
        self.saveFolderFrame.grid( row=0,column=1,sticky='nsew' )


    #def setLesionFrame( self, ):
    #    self.lesionFrame = tk.Frame( root,bg='#5C5C5C',borderwidth=15,relief='ridge' )
    #    lesionLabel = tk.Label( self.lesionFrame,text='Number of Lesions',anchor='center' )
    #    lesionLabel.grid( row=0,column=0,sticky=''  )

    #    self.numberOfLesionsEntry = tk.Entry( self.lesionFrame )
    #    self.numberOfLesionsEntry.insert( END,'4' )
    #    self.numberOfLesionsEntry.grid( row=1,column=0,sticky='' )

    #    self.lesionFrame.grid_rowconfigure( (0,1),weight=1 )
    #    self.lesionFrame.grid_columnconfigure( 0,weight=1 )
    #    self.lesionFrame.grid( row=1,column=1,sticky='nsew' )

    def setLesionFrame( self,bg,fg ):
        self.lesionFrame = tk.Frame( root,bg=bg,borderwidth=10,relief='ridge' )
        lesionLabel = tk.Label( self.lesionFrame,text='Num Lesions',anchor='center',bg=bg )
        #lesionLabel.grid( row=0,column=0,sticky='ew'  )
        lesionLabel.pack( side='left' )

        self.numberOfLesionsEntry = tk.Entry( self.lesionFrame,width=2 )
        self.numberOfLesionsEntry.insert( END,'4' )
        #self.numberOfLesionsEntry.grid( row=0,column=1,sticky='w' )
        self.numberOfLesionsEntry.pack( side='left' )

        #self.lesionFrame.grid_rowconfigure( (0),weight=1 )
        #self.lesionFrame.grid_columnconfigure( (0,1),weight=1 )
        self.lesionFrame.grid( row=2,column=0,sticky='nsew' )

    #def setInnocFrame( self, ):
    #    self.innocFrame = tk.Frame( root,bg='#C0BA80',borderwidth=15,relief='ridge' )

    #    self.innocTitle = tk.Label( self.innocFrame,text='Set Innoculation start Time',anchor='center' )
    #    self.innocTitle.grid( columnspan=2,row=0,sticky=''  )

    #    self.innocYearLabel = tk.Label( self.innocFrame,text='Year :' )
    #    self.innocYearLabel.grid( row=1,column=0,sticky='e' )
    #    self.innocYearTimeVar = tk.Entry( self.innocFrame )
    #    self.innocYearTimeVar.insert( END,'2000' )
    #    self.innocYearTimeVar.grid( row=1,column=1,sticky='w' )

    #    self.innocMonthLabel = tk.Label( self.innocFrame,text='Month :' )
    #    self.innocMonthLabel.grid( row=2,column=0,sticky='e' )
    #    self.innocMonthTimeVar = tk.Entry( self.innocFrame )
    #    self.innocMonthTimeVar.insert( END,'2' )
    #    self.innocMonthTimeVar.grid( row=2,column=1,sticky='w' )

    #    self.innocDayLabel = tk.Label( self.innocFrame,text='Day :' )
    #    self.innocDayLabel.grid( row=3,column=0,sticky='e' )
    #    self.innocDayTimeVar = tk.Entry( self.innocFrame )
    #    self.innocDayTimeVar.insert( END,'2' )
    #    self.innocDayTimeVar.grid( row=3,column=1,sticky='w' )

    #    self.innocHourLabel = tk.Label( self.innocFrame,text='Hour :' )
    #    self.innocHourLabel.grid( row=4,column=0,sticky='e' )
    #    self.innocHourTimeVar = tk.Entry( self.innocFrame )
    #    self.innocHourTimeVar.insert( END,'2' )
    #    self.innocHourTimeVar.grid( row=4,column=1,sticky='w' )

    #    self.innocFrame.grid_rowconfigure( (0,1,2,3,4),weight=1 )
    #    self.innocFrame.grid_columnconfigure( (0,1),weight=1 )
    #    self.innocFrame.grid( row=0,rowspan=1,column=2,sticky='nsew' )

    def setInnocFrame( self,bg,fg ):
        self.innocFrame = tk.Frame( root,bg=bg,borderwidth=10,relief='ridge' )

        self.innocTimeLabel = tk.Label( self.innocFrame,text='Innoculation Time',anchor='center',bg=bg )
        self.innocTimeLabel.grid( row=0,column=0,columnspan=4,sticky='s' )
        
        self.innocYear = tk.Frame( self.innocFrame,bg=bg,borderwidth=0)
        self.innocYearLabel = tk.Label( self.innocYear,text='Year',bg=bg)
        self.innocYearLabel.grid( row=0,column=0,stick='s' )
        self.innocYearTimeVar = tk.Entry( self.innocYear,width=4 )
        self.innocYearTimeVar.insert( END,'2000' )
        self.innocYearTimeVar.grid( row=1,column=0,sticky='e' )
        self.innocYear.grid( row=1,column=0,sticky='n' )

        self.innocMonth = tk.Frame( self.innocFrame,bg=bg,borderwidth=0)
        self.innocMonthLabel = tk.Label( self.innocMonth,text='Month',bg=bg)
        self.innocMonthLabel.grid( row=0,column=0,stick='' )
        self.innocMonthTimeVar = tk.Entry( self.innocMonth,width=2 )
        self.innocMonthTimeVar.insert( END,'2' )
        self.innocMonthTimeVar.grid( row=1,column=0,sticky='' )
        self.innocMonth.grid( row=1,column=1,sticky='n' )

        self.innocDay = tk.Frame( self.innocFrame,bg=bg,borderwidth=0)
        self.innocDayLabel = tk.Label( self.innocDay,text='Day',bg=bg)
        self.innocDayLabel.grid( row=0,column=0,stick='' )
        self.innocDayTimeVar = tk.Entry( self.innocDay,width=2 )
        self.innocDayTimeVar.insert( END,'2' )
        self.innocDayTimeVar.grid( row=1,column=0,sticky='' )
        self.innocDay.grid( row=1,column=2,sticky='n' )

        self.innocHour = tk.Frame( self.innocFrame,bg=bg,borderwidth=0)
        self.innocHourLabel = tk.Label( self.innocHour,text='Hour',bg=bg)
        self.innocHourLabel.grid( row=0,column=0,stick='' )
        self.innocHourTimeVar = tk.Entry( self.innocHour,width=2 )
        self.innocHourTimeVar.insert( END,'2' )
        self.innocHourTimeVar.grid( row=1,column=0,sticky='' )
        self.innocHour.grid( row=1,column=3,sticky='n' )

        self.innocFrame.grid_rowconfigure( (0,1),weight=1 )
        self.innocFrame.grid_columnconfigure( (0,1,2,3),weight=1 )
        self.innocFrame.grid( row=1,column=1,columnspan=2,sticky='nsew' )

    #def setMetaFrame( self, ):
    #        self.metaFrame = tk.Frame( root,bg='#C0BA80',borderwidth=15,relief='ridge' )

    #        self.metaFrame.grid_rowconfigure( (0,1,2),weight=1 )
    #        self.metaFrame.grid_columnconfigure( 0,weight=1 )
    #        self.metaFrame.grid( row=1,column=2,sticky='nsew' )

    #        self.selectmetafile = tk.Button( self.metaFrame,text='Select Experimental Design File',command=self.setMetaFile )
    #        self.selectmetafile.grid( row=0,column=0,sticky='' )

    #        self.saveMetaLabel = tk.Label( self.metaFrame,text='Experimental Design File:' )
    #        self.saveMetaLabel.grid( row=1,column=0,sticky='s' )

    #        self.metaFrame.update()
    #        self.metaFile = 'None'
    #        self.metaFileVar = tk.StringVar()
    #        self.metaFileLabel = tk.Label( self.metaFrame,textvariable=self.metaFileVar,wraplength=int(0.9*(self.metaFrame.winfo_width())) )
    #        self.metaFileLabel.grid( row=3,column=0,sticky='n' )

    def setDirFrame( self,bg,fg ):
        self.dirFrame = tk.Frame( root,bg=bg,borderwidth=10,relief='ridge' )
        self.dirFrameFG = fg

        self.selectsavedir = tk.Button( self.dirFrame,text='Result Dir',command=self.setSaveDir, bg=fg )
        self.selectsavedir.grid( row=0,column=0,sticky='' )

        self.selectmetafile = tk.Button( self.dirFrame,text='Meta-Data',command=self.setMetaFile, bg=fg )
        self.selectmetafile.grid( row=1,column=0,sticky='' )

        self.dirFrame.update()
        self.metaFile = 'None'
        self.metaFileVar = tk.StringVar()
        self.saveDir = 'None'
        self.saveDirVar = tk.StringVar()

        self.dirFrame.grid_rowconfigure( (0,1),weight=1 )
        self.dirFrame.grid_columnconfigure( 0,weight=1 )
        self.dirFrame.grid( row=0,column=2,sticky='nsew' )

    #def setDescFrame( self, ):
    #    self.descFrame = tk.Frame( root,bg='#FDC47D',borderwidth=15,relief='ridge' )

    #    descLabel = tk.Label( self.descFrame,text="Description of this experiment and image series",anchor='center' )
    #    descLabel.grid( column=1,row=0,sticky='' )

    #    self.e1 = tk.Text( self.descFrame,height=4,width=20,font=('Helvetica',24) )
    #    self.e1.grid( column=1,row=1,rowspan=2,sticky='nsew' )

    #    self.savebutton()

    #    self.descFrame.grid_rowconfigure( (0,1,2,3),weight=1 )
    #    self.descFrame.grid_columnconfigure( (0,2),weight=1 )
    #    self.descFrame.grid_columnconfigure( 1,weight=3 )
    #    self.descFrame.grid( row=0,column=3,sticky='nsew' )

    def setDescFrame( self,bg,fg ):
        self.descFrame = tk.Frame( root,bg=bg,borderwidth=10,relief='ridge' )

        descLabel = tk.Label( self.descFrame,text="Description:",anchor='center',bg=bg )
        #descLabel.grid( column=0,row=0,sticky='we' )
        descLabel.pack( side='left' )

        self.e1 = tk.Entry( self.descFrame,width=20 )
        #self.e1.grid( row=0,column=1,sticky='w' )
        self.e1.pack( side='left' )

        #self.descFrame.grid_rowconfigure( (0),weight=1 )
        #self.descFrame.grid_columnconfigure( 0,weight=1 )
        self.descFrame.grid( row=2,column=1,columnspan=2,sticky='nsew' )

    #def setRunFrame( self, ):
    #    self.runFrame = tk.Frame( root,bg='#EA3B46',borderwidth=15,relief='ridge' )

    #    self.gpuNumLabel = tk.Label( self.runFrame,text='Select Which GPU\nOr -1 For CPU' )
    #    self.gpuNumLabel.grid( row=0,column=0,sticky='e' )
    #    self.gpuNumEntry = tk.Entry( self.runFrame,justify='center' )
    #    self.gpuNumEntry.insert( END,'-1' )
    #    self.gpuNumEntry.grid( row=0,column=1,sticky='w' )

    #    self.Runbutton = tk.Button( self.runFrame,text='Run',command=self.process )
    #    self.Runbutton.grid( row=1,columnspan=2 )

    #    self.runFrame.grid_rowconfigure( (0,1),weight=1 )
    #    self.runFrame.grid_columnconfigure( (0,1),weight=1 )
    #    self.runFrame.grid( row=1,column=3,sticky='nsew' )

    def setRunFrame( self,bg,fg ):
        self.runFrame = tk.Frame( root,bg=bg,borderwidth=10,relief='ridge' )

        self.Runbutton = tk.Button( self.runFrame,text='Run',command=self.process,bg=fg )
        self.Runbutton.grid( row=0,column=0 )

        self.runFrame.grid_rowconfigure( 0,weight=1 )
        self.runFrame.grid_columnconfigure( 0,weight=1 )
        self.runFrame.grid( row=1,rowspan=2,column=3,sticky='nsew' )

    def setSegmentationFrame( self,bg,fg ):
        self.segmentationFrame = tk.Frame( root,bg=bg,borderwidth=10,relief='ridge' )
        self.segmentationFrameFG = fg

        self.segmentationLabel = tk.Label(self.segmentationFrame,text='Seg Results',bg=bg)
        self.segmentationLabel.grid(row=0,column=0,sticky='')

        self.segmentationFrame.grid_rowconfigure( (0),weight=1 )
        self.segmentationFrame.grid_columnconfigure( (0),weight=1 )
        self.segmentationFrame.grid( row=0,column=3,sticky='nsew' )

    def openSeg(self, segLst):
        global seg_current
        seg_current = 0
        fg = self.segmentationFrameFG

        self.original_seg = Image.open( segLst[0] )
        self.image_seg = ImageTk.PhotoImage( self.original )
        self.canvas_seg = Canvas( self.segmentationFrame,bd=0,highlightthickness=0 )
        self.canvasArea_seg = self.canvas.create_image( 0,0,image=self.image_seg,anchor=NE )
        self.canvas_seg.grid( row=0,columnspan=2,sticky=W+E+N+S)
        self.canvas_seg.bind( "<Configure>",self.resizeSeg )

        self.previousbuttonSeg = tk.Button( self.segmentationFrame,text='<-',command=lambda: self.move(-1),bg=fg )
        self.nextbuttonSeg = tk.Button( self.segmentationFrame,text='->',command=lambda: self.move(+1),bg=fg )

        self.segmentationFrame.update()

    ## gets models from the pytorch_models directory
    #def getModelsAvailable( self, ):
    #    models_available = glob.glob( 'pytorch_models/*' )
    #    models_available_list = []
    #    for m in models_available:
    #        models_available_list.append( m.split('/')[-1] )

    #    modelTypeLabel = tk.Label( self.modelFrame,text='Select Model to Use',anchor='center' )
    #    modelTypeLabel.grid( column=0,row=0,sticky=''  )

    #    return models_available_list

    # gets models from the pytorch_models directory
    def getModelsAvailable( self, ):
        models_available = glob.glob( 'pytorch_models/*' )
        models_available_list = []
        for m in models_available:
            models_available_list.append( m.split('/')[-1].split('\\')[-1].replace('.pth','') )

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

        #description = self.e1.get('1.0', END)
        description = self.e1.get()
        with open(description_pth, 'a') as file:
            file.write(str(description) + '\n')
        print('\nDescription saved')

    def savebutton( self, ):
        self.savebutton = tk.Button( self.descFrame,text='save',command=self.save,anchor='center' )
        self.savebutton.grid( column=1,row=3 )

    def selectbutton( self,fg ):
        self.imageFrameFG = fg
        self.selectbutton = tk.Button( self.imageFrame,text='Select Images', command=self.openimage, bg=fg )
        #self.selectbutton.grid( columnspan=2,row=0)
        self.selectbutton.grid( row=0,column=0 )

    #def selectsavedir( self, ):
    #    self.selectsavedir = tk.Button( self.dirFrame,text='Set Results Directory',command=self.setSaveDir )
    #    self.selectsavedir.grid( row=3,column=0,sticky='' )

    #    self.saveDirLabel0 = tk.Label( self.dirFrame,text='Results Directory:' )
    #    self.saveDirLabel0.grid( row=4,column=0,sticky='s' )

    #    self.dirFrame.update()
    #    self.saveDir = 'None'
    #    self.saveDirVar = tk.StringVar()
    #    self.saveDirLabel = tk.Label( self.dirFrame,textvariable=self.saveDirVar,wraplength=int(0.9*(self.dirFrame.winfo_width())) )
    #    self.saveDirLabel.grid( row=5,column=0,sticky='n' )

    #def selectsavedir( self, ):
    #    self.selectsavedir = tk.Button( self.dirFrame,text='Set Results Directory',command=self.setSaveDir )
    #    self.selectsavedir.grid( row=3,column=0,sticky='' )

    #    #self.saveDirLabel0 = tk.Label( self.dirFrame,text='Results Directory:' )
    #    #self.saveDirLabel0.grid( row=4,column=0,sticky='s' )

    #    self.dirFrame.update()
    #    self.saveDir = 'None'
    #    self.saveDirVar = tk.StringVar()
    #    #self.saveDirLabel = tk.Label( self.dirFrame,textvariable=self.saveDirVar,wraplength=int(0.9*(self.dirFrame.winfo_width())) )
    #    #self.saveDirLabel.grid( row=5,column=0,sticky='n' )

    #def openimage(self):
    #    global imagelst
    #    global current
    #    current = 0

    #    self.filenames = filedialog.askopenfilenames(filetypes=( ('all files','*.*'),('png files','*.png'),('jpeg files','*.jpeg') ), initialdir='/', title='Select Image')
    #    imagelst = self.sortimages( self.filenames )

    #    self.imageNameLabel = tk.Label( self.imageFrame )
    #    self.original = Image.open( imagelst[0] )
    #    self.image = ImageTk.PhotoImage( self.original )
    #    self.canvas = Canvas( self.imageFrame,bd=0,highlightthickness=0 )
    #    self.canvasArea = self.canvas.create_image( 0,0,image=self.image,anchor=NW )
    #    self.canvas.grid( row=2,columnspan=2,sticky=W+E+N+S)
    #    self.canvas.bind( "<Configure>",self.resize )

    #    self.imageFrame.update()
    #    self.imageNameVar = tk.StringVar()
    #    self.imageNameVar.set( imagelst[ current ] )
    #    self.imageNameLabel = tk.Label( self.imageFrame,textvariable=self.imageNameVar,wraplength=int(0.9*(self.canvas.winfo_width())) )
    #    self.imageNameLabel.grid( row=1,columnspan=2,sticky='s' )

    def openimage(self):
        global imagelst
        global current
        current = 0
        fg = self.imageFrameFG

        self.filenames = filedialog.askopenfilenames(filetypes=( ('all files','*.*'),('png files','*.png'),('jpeg files','*.jpeg') ), initialdir='/', title='Select Image')
        imagelst = self.sortimages( self.filenames )

        #self.imageNameLabel = tk.Label( self.imageFrame )
        self.original = Image.open( imagelst[0] )
        self.image = ImageTk.PhotoImage( self.original )
        self.canvas = Canvas( self.imageFrame,bd=0,highlightthickness=0 )
        self.canvasArea = self.canvas.create_image( 0,0,image=self.image,anchor=NW )
        self.canvas.grid( row=0,column=0,sticky=W+E+N+S )
        self.canvas.bind( "<Configure>",self.resize )

        self.previousbutton = tk.Button( self.imageFrame,text='<-',width=2,command=lambda: self.move(-1),anchor='sw',bg=fg )
        self.nextbutton = tk.Button( self.imageFrame,text='->',width=2,command=lambda: self.move(+1),anchor='se',bg=fg )

        self.imageFrame.update()
        #self.imageNameVar = tk.StringVar()
        #self.imageNameVar.set( imagelst[ current ] )
        #self.imageNameLabel = tk.Label( self.imageFrame,textvariable=self.imageNameVar,wraplength=int(0.9*(self.canvas.winfo_width())) )
        #self.imageNameLabel.grid( row=1,columnspan=2,sticky='s' )

    def resize( self,event ):
        size = (event.width, event.height)
        resized = self.original.resize( size,Image.ANTIALIAS )
        self.image = ImageTk.PhotoImage( resized )
        self.canvas.create_image( 0,0,image=self.image,anchor=NW )

        canvas_height = self.canvas.winfo_height()
        canvas_width = self.canvas.winfo_width()

        self.previousbutton.place(x=0,y=canvas_height,anchor='sw')
        self.nextbutton.place(x=canvas_width,y=canvas_height,anchor='se')

        self.imageFrame.update()

    def resizeSeg( self,event_seg ):
        size_seg = (event_seg.width, event_seg.height)
        resized_seg = self.original_seg.resize( size_seg,Image.ANTIALIAS )
        self.image_seg = ImageTk.PhotoImage( resized_seg )
        self.canvas_seg.create_image( 0,0,image=self.image_seg,anchor=NW )

        canvas_height_seg = self.canvas_seg.winfo_height()
        canvas_width_seg = self.canvas_seg.winfo_width()

        self.previousbuttonSeg.place(x=0,y=canvas_height_seg,anchor='sw')
        self.nextbuttonSeg.place(x=canvas_width_seg,y=canvas_height_seg,anchor='se')
        self.segmentationFrame.update()

    def move(self, delta):
        global current

        current += delta
        if current < 0:
            current = len(imagelst)-1
        if current >= len(imagelst):
            current = 0
        #self.imageNameVar.set( imagelst[ current ] )

        self.original = Image.open( imagelst[current] )
        resized = self.original.resize( (self.canvas.winfo_width(),self.canvas.winfo_height()),Image.ANTIALIAS )
        self.image = ImageTk.PhotoImage( resized )
        self.canvas.itemconfig( self.canvasArea,image=self.image )
        self.imageFrame.update()

    def moveSeg(self, delta):
        global seg_current

        seg_current += delta
        if seg_current < 0:
            seg_current = len(imagelst)-1
        if seg_current >= len(imagelst):
            seg_current = 0
        #self.imageNameVar.set( imagelst[ seg_current ] )

        self.original_seg = Image.open( imagelst[seg_current] )
        resized = self.original_seg.resize( (self.canvas_seg.winfo_width(),self.canvas_seg.winfo_height()),Image.ANTIALIAS )
        self.image_seg = ImageTk.PhotoImage( resized )
        self.canvas_seg.itemconfig( self.canvasArea_seg,image=self.image_seg )
        self.segmentationFrame.update()


    #def setSaveDir(self):
    #    self.saveDir = filedialog.askdirectory( initialdir='/',title='Select Save Directory' )
    #    self.saveDirVar.set( self.saveDir )

    def setSaveDir(self):
        fg = self.dirFrameFG
        self.saveDir = filedialog.askdirectory( initialdir='/',title='Select Save Directory' )
        self.saveDirVar.set( self.saveDir )
        savedir_label = self.saveDir.split('\\')[-1].split('/')[-1]

        self.selectsavedir.destroy()
        self.savedirset = tk.Label( self.dirFrame,text=savedir_label,bg=fg,wraplength=int(0.9*(self.dirFrame.winfo_width())) )
        self.savedirset.grid( row=0,column=0,sticky='' )

    #def setMetaFile(self):
    #    self.metaFile = filedialog.askopenfilename( filetypes=( ('excel files','*.xlsx'),('csv files','*.csv'),('all files','*.*') ), initialdir='/', title='Select Experimental Design File')
    #    self.metaFileVar.set( self.metaFile )

    def setMetaFile(self):
        fg = self.dirFrameFG
        self.metaFile = filedialog.askopenfilename( filetypes=( ('excel files','*.xlsx'),('csv files','*.csv'),('all files','*.*') ), initialdir='/', title='Select Experimental Design File')
        self.metaFileVar.set( self.metaFile )
        metafile_label = self.metaFile.split('\\')[-1].split('/')[-1].strip('.xlsx')

        self.selectmetafile.destroy()
        self.selectmetaset = tk.Label( self.dirFrame,text=metafile_label,bg=fg,wraplength=int(0.9*(self.dirFrame.winfo_width())) )
        self.selectmetaset.grid( row=1,column=0,sticky='' )

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
            #TODO set to num_lesions
            for j in range(4):
                imagemat.append([imname, i, cameraID, year, month, day, hour])

        self.imageDF = pd.DataFrame( imagemat,columns=['Image Name','File Location','CameraID','Year','Month','Day','Hour'] )
        self.imageDF['Lesion #'] = ''
        self.imageDF['Lesion Area Pixels'] = ''
        self.imageDF['Adjusted Lesion Pixels'] = ''
        self.imageDF['Camera #'] = self.imageDF['CameraID']
        self.imageDF = self.imageDF.sort_values( by=['CameraID','Year','Month','Day','Hour'] )
        self.imageDF['image_id'] = np.arange( len(self.imageDF) )
        self.imageDF = self.imageDF.reset_index(drop=True)
        imagelst = self.imageDF['File Location']
        return imagelst

    # Formats the original dataframe based off of the images selected
    def formatDF( self, ):
        #description = self.e1.get('1.0', END)
        description = self.e1.get()
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

        n = int( -1 )
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

        imgsWLesions_dir = os.path.join( result_dir,'imgsWLesions' )
        if not os.path.exists(imgsWLesions_dir):
            os.makedirs(imgsWLesions_dir)

        model_dir = os.path.join( 'pytorch_models' )
        if not os.path.exists( model_dir ):
            os.makedirs( model_dir )

        self.formatDF()

        print('self.metaFile :',self.metaFile)
        self.readMetaFile()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print( '\tUsing device:',device )
        if device.type == 'cuda':
            print( '\t',torch.cuda.get_device_name(0) )


        # choose pytorch model based off of model type
        NUM_CLASSES = 2
        model_pth = os.path.join( model_dir,( self.modelTypeVar.get() ) )
        model_pth = model_pth + ".pth"
        #model = u_netDICE_Brown.UNetDICE(NUM_CLASSES)
        model = u_net.UNet(NUM_CLASSES)
        model = nn.DataParallel(model)
        model.to(device)
        #saved_state_dict = torch.load(model_pth, map_location=lambda storage, loc: storage)
        saved_state_dict = torch.load(model_pth)
        print("\nModel restored from:", model_pth)

        #model.load_state_dict(saved_state_dict['state_dict'])
        model.load_state_dict(saved_state_dict)
        model.eval()

        camera_ids = np.unique(self.imageDF['CameraID'])
        for cams_completed,camera_id in enumerate(camera_ids):

            cameraDF = self.imageDF.loc[lambda df: df['CameraID']==camera_id, : ].reset_index(drop=True)
            new_camera = True

            leaf_seg_stack = []
            leaf_img_stack = []
            leaf_mask = None
            row_mid = 0
            col_mid = 0
            half_side = 0
            resize_ratio = 0

            for df_index,df_row in cameraDF.iterrows():
                if df_index % self.num_lesions != 0:
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
                    image_tensor = torch.from_numpy(np.expand_dims(formatted_img,axis=0))
                    output = model(image_tensor).to(device)

                    if len(output.shape) > 3:
                        output = torch.argmax(output,axis=1)

                    msk = torch.squeeze(output).data.cpu().numpy()
                    #msk = np.where(msk>0,1,0)

                leaf_seg_stack.append(msk)
                leaf_img_stack.append(resized_image)
                for k in range(self.num_lesions):
                    cameraDF.at[ df_index+k,'ResizeRatio' ] = resize_ratio

            print('\nPostProcessing now!')
            label_map_ws = postp.watershedSegStack(np.array(leaf_seg_stack),self.num_lesions,postprocess_dir,camera_id)
            cameraDF = postp.processSegStack(np.array(leaf_seg_stack),leaf_img_stack,self.num_lesions,label_map_ws,cameraDF,resize_ratio,postprocess_dir,imgsWLesions_dir)

            clean_df = postp.cleanDF(cameraDF)
            clean_df.to_csv( result_file,mode='a',header=not os.path.exists(result_file),index=False )
            print( '\tresult_file located :',result_file )

        print( '\n****************************************************************************' )
        print( '***********************************DONE*************************************' )
        print( '****************************************************************************' )

        segLst = glob.glob(imgsWLesions_dir + '/*lesions.png')
        self.openSeg(segLst=segLst)



if __name__ == "__main__":
    root = tk.Tk()
    GUI(root)
    root.mainloop()
