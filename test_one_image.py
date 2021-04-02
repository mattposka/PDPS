# combine all the testing pipeline into a single file
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
def rmbackground(slide, MatPath, NewMatPath, ref_extent, ref_area):
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


if __name__ == "__main__":
    # input
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tumorname = "green"
    test_img_pth = "/data/AutoPheno/imgs_all/rapa_cam_33_2019-08-11_0500original.png"
    # setting
    patch_size = 512
    INPUT_SIZE = (patch_size, patch_size)
    overlap_size = 64
    ref_area = 10000  # pre-processing
    ref_extent = 0.6  # pre-processing
    rm_rf_area = 5000  # post-processing
    ref_ecc = 0.92  # post-processing
    BATCH_SIZE = 32
    model_id = '10700'  # 'BACH_UNET_B2_S110000_Frozen_BN_test2048'
    RESTORE_FROM = '/data/AutoPheno/green/200527/PatchNet/snapshots-fb/LEAF_UNET_B0064_S010700.pth'
    root_pth = '/workspace/data/AutoPheno/temp/'  # may change to './' in the final version
    savepath = root_pth + tumorname + '/512_test_stride_64'
    logdirpth = root_pth + tumorname + '/log'
    if not os.path.exists(logdirpth):
        os.makedirs(logdirpth)
    logpath = root_pth + tumorname + '/log/512_test_stride_64_XY3c.log'
    images = os.path.join(savepath, 'images')
    if not os.path.exists(images):
        os.makedirs(images)
    log = open(logpath, 'w')
    log.write(test_img_pth + '\n')
    filename = test_img_pth.split('/')[-1]
    slidename = filename[:-4]
    # divide testing image
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
    SavePatchMap(MatPath, FigPath, slidename+"_Map.npz")
    pred_img_pth = os.path.join(FigPath, slidename+"_Map.png")
    # crf_img_name = slidename+"_Map.png"
    # crf_img_pth = os.path.join(CRFFigPath, crf_img_name)
    # CRFs(test_img_pth, pred_img_pth, crf_img_pth)
    post_img_pth = os.path.join(PostPath, slidename+"_Map.png")
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
