# Outputing the entire heatmap and mask of LEAF
# Author: Haomiao Ni
import os
from scipy.sparse import load_npz, save_npz, csr_matrix
from scipy.misc import imread, imsave
from PIL import Image
import numpy as np
from utils.transforms import vl2im
from utils.mk_datasets import test_imgs_list
from skimage import filters
from skimage.measure import label,regionprops
from skimage.morphology import closing, square, remove_small_objects
from utils.postprocessing import postprocess
from utils.transforms import im2vl, vl2im
Image.MAX_IMAGE_PIXELS = 933120000


# remove background
def rmbackground(SlidePath, MatPath, NewMatPath, ref_extent, ref_area):
    SlideDir = os.listdir(SlidePath)
    SlideDir.sort()
    for SlideName in SlideDir:
        # if not "original" in SlideName:
        #     continue
        # if not SlideName in test_imgs_list:
        #     continue
        print(SlideName)
        slide = os.path.join(SlidePath, SlideName)
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

        MatName = SlideName.replace(".png", "_Map.npz")
        Mat = os.path.join(MatPath, MatName)
        SegRes = load_npz(Mat)
        SegRes = SegRes.todense()
        SegRes[low_s_bin] = 0
        save_npz(os.path.join(NewMatPath, MatName), csr_matrix(SegRes))


def SavePatchMap(MatPath, FigPath):
    MatList = os.listdir(MatPath)
    for MatName in MatList:
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


def SavePatchFigMap(OriginalPath, FigPath, FigMapPath):
    ImageList = os.listdir(OriginalPath)
    for ImageName in ImageList:
        # if not "original" in ImageName:
        #     continue
        # if not ImageName in test_imgs_list:
        #     continue
        FigMapName = ImageName.replace('.png', '_FIGMAP.png')
        FigMapFile = os.path.join(FigMapPath, FigMapName)
        print("processing:", ImageName)
        ImageFile = os.path.join(OriginalPath, ImageName)
        ImageArr = imread(ImageFile)
        (h, w, _) = ImageArr.shape
        ImageArr = ImageArr[0:h, 0:w, :]
        Img = Image.fromarray(ImageArr)
        FigName = ImageName.replace('.png', '_Map.png')
        FigFile = os.path.join(FigPath, FigName)
        Fig = Image.open(FigFile)

        # splicing
        if w < h:
            new_im = Image.new('RGB', (w*2, h))
            new_im.paste(Img, (0, 0))
            new_im.paste(Fig, (w, 0))
            new_im.save(FigMapFile)
        else:
            new_im = Image.new('RGB', (w, h*2))
            new_im.paste(Img, (0, 0))
            new_im.paste(Fig, (0, h))
            new_im.save(FigMapFile)


def SavePatchFigMapMask(OriginalPath, FigPath, MskPath, FigMapMskPath):
    ImageList = os.listdir(OriginalPath)
    for ImageName in ImageList:
        # if not "original" in ImageName:
        #     continue
        # if not ImageName in test_imgs_list:
        #     continue
        FigMapName = ImageName.replace('.png', '_FIGMAPMSK.png')
        FigMapFile = os.path.join(FigMapMskPath, FigMapName)
        print("processing:", ImageName)
        ImageFile = os.path.join(OriginalPath, ImageName)
        ImageArr = imread(ImageFile)
        (h, w, _) = ImageArr.shape
        ImageArr = ImageArr[0:h, 0:w, :]
        Img = Image.fromarray(ImageArr)
        FigName = ImageName.replace('.png', '_Map.png')
        FigFile = os.path.join(FigPath, FigName)
        Fig = Image.open(FigFile)

        MskName = ImageName.replace('original', tumorname)
        MskFile = os.path.join(MskPath, MskName)
        try:
            Msk = Image.open(MskFile)
        except:
            print(MskName + " is missing!")
        Msk = Msk.resize(Fig.size)
        # splicing
        if w < h:
            new_im = Image.new('RGB', (w * 3, h))
            new_im.paste(Img, (0, 0))
            new_im.paste(Fig, (w, 0))
            new_im.paste(Msk, (w*2, 0))
            new_im.save(FigMapFile)
        else:
            new_im = Image.new('RGB', (w, h * 3))
            new_im.paste(Img, (0, 0))
            new_im.paste(Fig, (0, h))
            new_im.paste(Msk, (0, h*2))
            new_im.save(FigMapFile)


def postprocess_rmsl(predicted_image_dir_path, new_image_path, ref_area, ref_ecc):
    if not os.path.exists(new_image_path):
        os.makedirs(new_image_path)
    for pth, dir, filenames in os.walk(predicted_image_dir_path):
        for filename in filenames:
            img_pth = os.path.join(pth, filename)
            img = imread(img_pth)
            img = im2vl(img)
            img_close = closing(img, square(3))
            labeled_img = label(img_close, connectivity=2)
            new_labeled_img = remove_small_objects(labeled_img, min_size=ref_area, connectivity=2)
            # remove non-circle region
            props = regionprops(new_labeled_img)
            for i, reg in enumerate(props):
                if reg.eccentricity > ref_ecc:
                    new_labeled_img[new_labeled_img == reg.label] = 0
            new_img = np.asarray(new_labeled_img != 0, dtype=np.uint8)
            new_img = vl2im(new_img)
            new_name = os.path.join(new_image_path, filename)
            imsave(new_name, new_img)


def SavePatchFigMapMaskRm(OriginalPath, FigPath, PostFigPath, MskPath, FigMapMskPath):
    ImageList = os.listdir(OriginalPath)
    for ImageName in ImageList:
        # if not "original" in ImageName:
        #     continue
        # if not ImageName in test_imgs_list:
        #     continue
        FigMapName = ImageName.replace('.png', '_FIGMAPMSK.png')
        FigMapFile = os.path.join(FigMapMskPath, FigMapName)
        print("processing:", ImageName)
        ImageFile = os.path.join(OriginalPath, ImageName)
        ImageArr = imread(ImageFile)
        (h, w, _) = ImageArr.shape
        ImageArr = ImageArr[0:h, 0:w, :]
        Img = Image.fromarray(ImageArr)
        FigName = ImageName.replace('.png', '_Map.png')
        FigFile = os.path.join(FigPath, FigName)
        Fig = Image.open(FigFile)

        PostFile = os.path.join(PostFigPath, FigName)
        Post = Image.open(PostFile)

        MskName = ImageName.replace('original', tumorname)
        MskFile = os.path.join(MskPath, MskName)
        try:
            Msk = Image.open(MskFile)
        except:
            print(MskName + " is missing!")
        Msk = Msk.resize(Fig.size)
        # splicing
        if w < h:
            new_im = Image.new('RGB', (w * 4, h))
            new_im.paste(Img, (0, 0))
            new_im.paste(Fig, (w, 0))
            new_im.paste(Post, (w*2, 0))
            new_im.paste(Msk, (w*3, 0))
            new_im.save(FigMapFile)
        else:
            new_im = Image.new('RGB', (w, h * 4))
            new_im.paste(Img, (0, 0))
            new_im.paste(Fig, (0, h))
            new_im.paste(Post, (0, h*2))
            new_im.paste(Msk, (0, h*3))
            new_im.save(FigMapFile)


if __name__ == '__main__':
    SrcPath = '/data/AutoPheno/'
    tumorname = "green"
    version = "200717"
    modelname = "10700"
    postfix = "-fb"
    slidedir = "random_test_imgs"
    ref_extent = 0.6
    ref_area = 10000
    # OldMatPath: the npz result generated by running marge_npz_final.py.
    OldMatPath = os.path.join(SrcPath, tumorname, version, 'PatchNet', 'npz'+postfix, 'whole_npz', modelname)
    # SrcPath + tumorname + '/' + 'PatchNet/npz/whole_npz/1800'
    MatPath = os.path.join(SrcPath, tumorname, version, 'PatchNet', 'npz'+postfix, 'whole_npz_post', modelname)
    # SrcPath + tumorname + '/' + 'PatchNet/npz/whole_npz_post/1800'
    FigPath = os.path.join(SrcPath, tumorname, version, 'PatchNet', 'npz'+postfix, 'whole_fig_post', modelname)
    # SrcPath + tumorname + '/' + 'PatchNet/npz/whole_fig_post/1800'
    CRFFigPath = os.path.join(SrcPath, tumorname, version, 'PatchNet', 'npz'+postfix, 'CRF')
    # "/data/AutoPheno/"+tumorname+"/PatchNet/npz/CRF"
    RemoveSmallFigPath = os.path.join(SrcPath, tumorname, version, 'PatchNet', 'npz'+postfix, 'whole_fig_re_small', modelname)
    MskPath = SrcPath + slidedir
    SlidePath = SrcPath + slidedir
    FigMapPath = os.path.join(SrcPath, tumorname, version, 'PatchNet', 'npz'+postfix, 'whole_figmap', modelname)
    # SrcPath + tumorname + '/' + 'PatchNet/npz/whole_figmap/1800'
    FigMapMskPath = os.path.join(SrcPath, tumorname, version, 'PatchNet', 'npz'+postfix, 'whole_figmapmsk', modelname)
    # SrcPath + tumorname + '/' + 'PatchNet/npz/whole_figmapmsk/1800'
    FigOriMapMskPath = os.path.join(SrcPath, tumorname, version, 'PatchNet', 'npz'+postfix, 'whole_figorimapmsk', modelname)
    FigOriMapMskPostPath = os.path.join(SrcPath, tumorname, version, 'PatchNet', 'npz'+postfix, 'whole_figorimapmskpost', modelname)
    if not os.path.exists(MatPath):
        os.makedirs(MatPath)
    if not os.path.exists(FigPath):
        os.makedirs(FigPath)
    if not os.path.exists(FigMapPath):
        os.makedirs(FigMapPath)
    if not os.path.exists(FigMapMskPath):
        os.makedirs(FigMapMskPath)
    if not os.path.exists(FigOriMapMskPath):
        os.makedirs(FigOriMapMskPath)
    if not os.path.exists(FigOriMapMskPostPath):
        os.makedirs(FigOriMapMskPostPath)
    print("remove background")
    # rmbackground(SlidePath, OldMatPath, MatPath, ref_extent=ref_extent, ref_area=ref_area)
    print("saving segmentation results")
    # SavePatchMap(MatPath, FigPath)
    # print("saving segmentation results after CRF processing")
    # postprocess(SlidePath, FigPath, CRFFigPath)
    # print("saving original image and its segmentation results")
    # SavePatchFigMap(SlidePath, CRFFigPath, FigMapPath)
    # print("saving original image and its segmentation results and mask")
    # SavePatchFigMapMask(SlidePath, CRFFigPath, MskPath, FigMapMskPath)
    # SavePatchFigMapMask(SlidePath, FigPath, MskPath, FigOriMapMskPath)
    postprocess_rmsl(FigPath, RemoveSmallFigPath, 5000, 0.92)
    # SavePatchFigMapMaskRm(SlidePath, FigPath, RemoveSmallFigPath, MskPath, FigOriMapMskPostPath)
    # deal with new images without GT
    SavePatchFigMap(SlidePath, RemoveSmallFigPath, FigMapPath)


#rapa_cam_05_2019-08-03_1600green.png is missing!
# rapa-cam-05_2019-08-03_1600green.png
#rapa_cam_06_2019-08-02_1900green.png is missing!
# rapa_cam_06_08-02_1900green.png
#rapa_cam_31_2019-08-10_1600green.png is missing!
# rapa_cam_31_2019-08-09_1600green.png
