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
from PIL import Image
from utils.transforms import vl2im, im2vl
from skimage import filters
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects
from utils.postprocessing import CRFs
Image.MAX_IMAGE_PIXELS = 933120000
imagelst = []
current = 0
count = 0

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        print( '\nAverageMeter object made' )
        self.reset()

    def reset(self):
        print('AverageMeter.reset()')
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        print('AverageMeter.update()')
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# remove background
def rmbackground(slide, MatPath, NewMatPath, ref_extent, ref_area):
    print( '\nrmbackground called' )
    low_dim_img = Image.open(slide)
    low_hsv_img = low_dim_img.convert('HSV')
    _, low_s, _ = low_hsv_img.split()

    # --OSTU threshold
    low_s_thre = filters.threshold_otsu(np.array(low_s))
    low_s_bin = low_s > low_s_thre  # row is y and col is x

    # only keep the largest connected component
    low_s_bin = closing(low_s_bin, square(3))
    labeled_img = label(low_s_bin, connectivity=2)

    props = regionprops(labeled_img)
    area_list = np.zeros(len(props))
    for i, reg in enumerate(props):  # i+1 is the label
        area_list[i] = reg.area
    # sort
    area_list = area_list.argsort()[::-1]
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

    assert label_id != -1, "fail to find the leaf region in pre-processing!" \
                           "try to REDUCE 'ref_extent' a bit"
    assert label_area > ref_area, "fail to find the leaf region in pre-processing!" \
                                  "try to REDUCE 'ref_extent' a bit"
    low_s_bin = labeled_img != label_id

    SlideName = slide.split('/')[-1]
    MatName = SlideName.replace(".png", "_Map.npz")
    Mat = os.path.join(MatPath, MatName)
    SegRes = load_npz(Mat)
    SegRes = SegRes.todense()
    SegRes[low_s_bin] = 0
    save_npz(os.path.join(NewMatPath, MatName), csr_matrix(SegRes))

def SavePatchMap(MatPath, FigPath, MatName):
    print( '\nSavePatchMap called' )
    FigName = MatName.replace('.npz', '.png')
    FigFile = os.path.join(FigPath, FigName)
    MatFile = os.path.join(MatPath, MatName)
    SegRes = load_npz(MatFile)
    SegRes = SegRes.todense()
    SegRes = vl2im(SegRes)
    Fig = Image.fromarray(SegRes.astype(dtype=np.uint8))
    del SegRes
    Fig.convert('RGB')
    Fig.save(FigFile, 'PNG')

def findN(cir_list):
    print( '\nfindN called' )
    N = 0
    for i in cir_list:
        if i[0] > 20:
            N += 1

    return N

def postprocess(cir_list, rect_list, N, imag, imagelst):
    print( '\npostprocess called' )
    # variables
    global count
    sought = [0, 0, 255]
    count += 1
    lesion = []
    area = []
    max_cir = []
    root_pth = './result/'
    tumorname = "green"
    CoordPath = root_pth + tumorname + "/PatchNet/npz/Coord"
    LesionArea = root_pth + tumorname + "/PatchNet/npz/Area"
    resultpath = root_pth + tumorname + "/PatchNet/npz/resultfiles/spreadsheets"

    max_rect = [p[1] for p in sorted(sorted(enumerate(rect_list), key=lambda x: x[1])[-N:], key=itemgetter(0))]
    lesion_id = [p[0]+1 for p in enumerate(max_rect)]
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
    print('bounding box coordinates saved')
    if not os.path.exists(LesionArea):
        os.makedirs(LesionArea)
    filename = 'lesion_area.txt'
    with open(os.path.join(LesionArea, filename), 'a') as file:
        file.write('image' + str(count) + ':' + str(area).strip('[]') + '\n')
    print('lesion area saved')

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
    print('circle center coordinates and radius saved')

    # spreadsheet
    imname = imagelst[0].split('/')[-1]
    cameraID = imname[9:11]
    year = imname[12:16]
    month = imname[17:19]
    day = imname[20:22]
    time = imname[23:27]

    l1 = cameraID
    l2 = year
    l3 = month
    l4 = day
    l5 = time
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
    filename = spreadsheet + '.xlsx'

    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    pp = os.path.join(resultpath, filename)

    if not os.path.exists(pp):
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(os.path.join(resultpath, filename), engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
    else:
        writer = pd.ExcelWriter(os.path.join(resultpath, filename), engine='openpyxl')
        # try to open an existing workbook
        writer.book = load_workbook(os.path.join(resultpath, filename))
        # copy existing sheets
        writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        # read existing file
        reader = pd.read_excel(os.path.join(resultpath, filename))
        # write out the new sheet
        df.to_excel(writer, index=False, header=False, startrow=len(reader) + 1)
        writer.close()

class GUI(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        print( '\nGUI object made')
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        parent.title('Coco Leaf Image Processor')
        parent.minsize(640, 400)

        tk.Label(text="Description of this experiment and image series").grid(column=3, row=0)

        self.entryboxes()
        self.savebutton()
        self.selectbutton()
        self.RunwithCPUbutton()
        self.RunwithGPUbutton()
        self.nextbutton()
        self.previousbutton()

    def entryboxes(self):
        self.e1 = tk.Text(root, height=18, width=40)
        self.e2 = tk.Entry(root)
        self.e1.grid(column=3, row=1)
        self.e2.grid(column=1, row=2)

    def save(self):
        root_pth = './result/'
        tumorname = "green"
        resultpath = root_pth + tumorname + "/PatchNet/npz/resultfiles/descriptions"
        a = self.e1.get('1.0', END)
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)
        filename = 'description.txt'
        with open(os.path.join(resultpath, filename), 'a') as file:
            file.write(str(a) + '\n')
        print('information saved')

    def savebutton(self):
        self.savebutton = tk.Button(text='save', command=self.save)
        self.savebutton.grid(column=3, row=2)

    def selectbutton(self):
        self.selectbutton = tk.Button(text='Select Images', command=self.openimage)
        self.selectbutton.grid(column=1, row=0)

    def RunwithCPUbutton(self):
        self.RunwithCPUbutton = tk.Button(text='Run with CPU', command=lambda: self.process(""))
        self.RunwithCPUbutton.grid(column=1, row=3)

    def RunwithGPUbutton(self):
        self.RunwithGPUbutton = tk.Button(text='Run with GPU', command=lambda: self.process(self.e2.get()), state=tk.DISABLED)
        self.RunwithGPUbutton.grid(column=1, row=4)

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
        self.nextbutton = tk.Button(text='Next', command=lambda: self.move(+1))
        self.nextbutton.grid(column=2, row=2)

    def previousbutton(self):
        self.previousbutton = tk.Button(text='Previous', command=lambda: self.move(-1))
        self.previousbutton.grid(column=0, row=2)

    def openimage(self):
        global imagelst
        global imglst
        self.filenames = filedialog.askopenfilenames(filetypes=(('jpeg files', '*.jpeg'), ('png files', '*.png'), ('all files', '*.*')), initialdir='/', title='Select Image')
        for i in self.filenames:
            imagelst.append(i)
        pic = Image.open(imagelst[0])
        pic = pic.resize((230, 230), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(pic)
        self.label = Label(image=img)
        self.label.image = img
        self.label.grid(column=1, row=1)

    def process(self, n):
        print( 'GUI.process start' )
        os.environ["CUDA_VISIBLE_DEVICES"] = n
        while imagelst:
            # input
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            tumorname = "green"
            test_img_pth = imagelst[0]
            # setting
            patch_size = 512
            INPUT_SIZE = (patch_size, patch_size)
            overlap_size = 64
            ref_area = 10000  # pre-processing
            ref_extent = 0.6  # pre-processing
            rm_rf_area = 5000  # post-processing
            ref_ecc = 0.92  # post-processing
            BATCH_SIZE = 32
            model_id = '21700'  # 'BACH_UNET_B2_S110000_Frozen_BN_test2048'
            ##################################################################################################################################################
            # look over this area
            ##################################################################################################################################################
            RESTORE_FROM = 'LEAF_UNET_B0064_S021700.pth'
            root_pth = './result/'  # may change to './' in the final version
            savepath = root_pth + tumorname + '/512_test_stride_64'
            print( 'savepath :',savepath )
            logdirpth = root_pth + tumorname + '/log'
            print( 'logdirpth :',logdirpth )
            if not os.path.exists(logdirpth):
                os.makedirs(logdirpth)
            logpath = root_pth + tumorname + '/log/512_test_stride_64_XY3c.log'
            print( 'logpath :',logpath )
            images = os.path.join(savepath, 'images')
            if not os.path.exists(images):
                os.makedirs(images)
            print( 'before logpath open()' )
            log = open(logpath, 'w')
            print( 'before log.write()' )
            log.write(test_img_pth + '\n')
            filename = test_img_pth.split('/')[-1]
            slidename = filename[:-4]
            # divide testing image
            print( 'divide testing image' )
            process_tif(test_img_pth, filename, images, log, patch_size,
                        overlap_size, ref_area=ref_area, ref_extent=ref_extent)
            # make testing dataset
            txtdirpth = os.path.join(root_pth, tumorname, 'txt')
            if not os.path.exists(txtdirpth):
                os.makedirs(txtdirpth)
            txtname = slidename + ".txt"
            txtfile = os.path.join(txtdirpth, txtname)
            txt = open(txtfile, 'w')
            SlideDir = os.path.join(images, slidename)
            PatchList = os.listdir(SlideDir)
            for PatchName in PatchList:
                PatchFile = os.path.join(SlideDir, PatchName)
                txt.write(PatchFile + '\n')
            txt.close()
            #################################################
            # testing
            IMG_MEAN = np.array((62.17962105572224, 100.62603236734867, 131.60830906033516), dtype=np.float32)
            DATA_DIRECTORY = images
            DATA_LIST_PATH = txtdirpth
            NUM_CLASSES = 2
            NPZ_PATH = root_pth + tumorname + '/PatchNet/npz/' + model_id
            if not os.path.exists(NPZ_PATH):
                os.makedirs(NPZ_PATH)
            MAP_PATH = root_pth + tumorname + '/PatchNet/map/' + model_id
            if not os.path.exists(MAP_PATH):
                os.makedirs(MAP_PATH)
            LOG_PATH = root_pth + tumorname + '/PatchNet/logfiles'
            if not os.path.exists(LOG_PATH):
                os.makedirs(LOG_PATH)
            preprocess_start_time = time.time()
            print("Restored from:", RESTORE_FROM)
            LogName = "Test_HeatMap_log.txt"
            LogFile = os.path.join(LOG_PATH, LogName)
            log = open(LogFile, 'w')
            log.writelines('batch size:' + str(BATCH_SIZE) + '\n')
            log.writelines(DATA_LIST_PATH + '\n')
            log.writelines('restore from ' + RESTORE_FROM + '\n')
            model = UNet(NUM_CLASSES)
            model = nn.DataParallel(model)
            model.to(device)
            saved_state_dict = torch.load(RESTORE_FROM, map_location=lambda storage, loc: storage)
            num_examples = saved_state_dict['example']
            print("using running mean and running var")
            log.writelines("using running mean and running var\n")
            model.load_state_dict(saved_state_dict['state_dict'])
            model.eval()
            log.writelines('preprocessing time: ' + str(time.time() - preprocess_start_time) + '\n')
            print('Processing ' + slidename)
            log.writelines('Processing ' + slidename + '\n')
            TestTxt = os.path.join(DATA_LIST_PATH, slidename + '.txt')
            testloader = data.DataLoader(LEAFTest(TestTxt, crop_size=INPUT_SIZE, mean=IMG_MEAN),
                                         batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
            TestNpzPath = os.path.join(NPZ_PATH, slidename)
            TestMapPath = os.path.join(MAP_PATH, slidename)
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

                    for ind in range(0, pred[0].size(0)):
                        prob = torch.squeeze(pred[0][ind]).data.cpu().numpy()
                        prob = coo_matrix(prob)
                        if len(prob.data) == 0:
                            continue
                        mapname = name[ind].replace('.jpg', '_N' + str(num_examples) + '_MAP.npz')
                        mapfile = os.path.join(TestMapPath, mapname)
                        save_npz(mapfile, prob.tocsr())

                        msk = torch.squeeze(pred[1][ind]).data.cpu().numpy()
                        msk = coo_matrix(msk)
                        if len(msk.data) == 0:
                            continue
                        npzname = name[ind].replace('.jpg', '_N' + str(num_examples) + '_MSK.npz')
                        npzfile = os.path.join(TestNpzPath, npzname)
                        save_npz(npzfile, msk.tocsr())

                    batch_time.update(time.time() - end)
                    end = time.time()

                    if index % 10 == 0:
                        print('Test:[{0}/{1}]\t'
                              'Time {batch_time.val:.3f}({batch_time.avg:.3f})'
                              .format(index, len(testloader), batch_time=batch_time))

            print('The total test time for ' + slidename + ' is ' + str(batch_time.sum))
            log.writelines('batch num:' + str(len(testloader)) + '\n')
            log.writelines('The total test time for ' + slidename + ' is ' + str(batch_time.sum) + '\n')
            # merge npz
            path = root_pth + tumorname + '/' + 'PatchNet/npz'
            savenpz = os.path.join(path, 'whole_npz/' + model_id)
            if not os.path.exists(savenpz):
                os.makedirs(savenpz)
            logdir = root_pth + tumorname + '/' + 'PatchNet/logfiles/'
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            logfile = open(os.path.join(logdir, 'merge_npz_' + model_id + '.log'), 'w')
            img = imread(test_img_pth)
            width, height = img.shape[1], img.shape[0]
            final = merge_npz(os.path.join(path, model_id), slidename,
                              os.path.join(savenpz, slidename + '_Map.npz'),
                              int(width), int(height))
            # saving whole heatmap and mask
            OldMatPath = root_pth + tumorname + '/' + 'PatchNet/npz/whole_npz/' + model_id
            MatPath = root_pth + tumorname + '/' + 'PatchNet/npz/whole_npz_post/' + model_id
            FigPath = root_pth + tumorname + '/' + 'PatchNet/npz/whole_fig_post/' + model_id
            # CRFFigPath = "/data/AutoPheno/" + tumorname + "/PatchNet/npz/CRF"
            PostPath = root_pth + tumorname + '/' + 'PatchNet/npz/whole_fig_post_rm/' + model_id
            if not os.path.exists(MatPath):
                os.makedirs(MatPath)
            if not os.path.exists(FigPath):
                os.makedirs(FigPath)
            if not os.path.exists(PostPath):
                os.makedirs(PostPath)
            # if not os.path.exists(CRFFigPath):
            #     os.makedirs(CRFFigPath)
            print("remove background")
            rmbackground(test_img_pth, OldMatPath, MatPath, ref_extent=ref_extent, ref_area=ref_area)
            SavePatchMap(MatPath, FigPath, slidename + "_Map.npz")
            pred_img_pth = os.path.join(FigPath, slidename + "_Map.png")
            # crf_img_name = slidename+"_Map.png"
            # crf_img_pth = os.path.join(CRFFigPath, crf_img_name)
            # CRFs(test_img_pth, pred_img_pth, crf_img_pth)
            post_img_pth = os.path.join(PostPath, slidename + "_Map.png")
            img = imread(pred_img_pth)
            img = im2vl(img)
            img_close = closing(img, square(3))
            labeled_img = label(img_close, connectivity=2)
            new_labeled_img = remove_small_objects(labeled_img, min_size=rm_rf_area, connectivity=2)
            # remove non-circle region
            props = regionprops(new_labeled_img)
            for i, reg in enumerate(props):
                if reg.eccentricity > ref_ecc:
                    new_labeled_img[new_labeled_img == reg.label] = 0
            new_img = np.asarray(new_labeled_img != 0, dtype=np.uint8)
            new_img = vl2im(new_img)
            imsave(post_img_pth, new_img)

            # Draw bounding box
            cv2.namedWindow(imagelst[0], cv2.WINDOW_NORMAL)
            imag = cv2.pyrDown(cv2.imread(post_img_pth, cv2.IMREAD_UNCHANGED))

            # threshold image
            ret, threshed_img = cv2.threshold(cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY),
                                              127, 255, cv2.THRESH_BINARY)
            # find contours and get the external one

            contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # with each contour, draw boundingRect in green
            # a minEnclosingCircle in blue
            rect_list = []
            cir_list = []

            for c in contours:
                # get the bounding rect
                x, y, w, h = cv2.boundingRect(c)
                # draw a green rectangle to visualize the bounding rect
                cv2.rectangle(imag, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rect_list.append([w * h, (x, y), (x + w, y + h)])

                # finally, get the min enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(c)
                # convert all values to int
                center = (int(x), int(y))
                radius = int(radius)
                # and draw the circle in blue
                imag = cv2.circle(imag, center, radius, (255, 0, 0), 2)
                cir_list.append([radius, center])

            postprocess(cir_list, rect_list, findN(cir_list), imag, imagelst)

            cv2.drawContours(imag, contours, -1, (255, 255, 0), 1)

            cv2.imshow(imagelst[0], imag)

            del imagelst[0]

if __name__ == "__main__":
    root = tk.Tk()
    print( 'root=tk.Tk() done' )
    GUI(root)
    print( 'GUI(root) done' )
    root.mainloop()
    print( 'root.mainloop() done' )

